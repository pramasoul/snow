# JAX GPU-Accelerated NEE Solver

## Overview

`snow.nlo_jax` provides a GPU-accelerated version of the Nonlinear Envelope
Equation (NEE) solver using [JAX](https://github.com/jax-ml/jax).  The entire
Dormand-Prince RK45 integration loop — including FFTs, upsampling, the
nonlinear product, and adaptive step control — is JIT-compiled into a single
fused GPU program.  There is zero Python dispatch overhead in the hot loop.

## Architecture

```
                     Python (setup)                    GPU (JIT-compiled)
                ┌─────────────────────┐          ┌──────────────────────────┐
   k(z) func ──>│ _build_poling_table │──> LUT ──│                          │
                │  (pre-sample at     │          │  jax.lax.while_loop:     │
   pulse,       │   many z-points)    │          │    fnl(z, y):            │
   waveguide ──>│                     │          │      exp(-j*D*z)         │
                │   Move arrays to    │──> A0 ──>│      FFT (upsample)      │
                │   JAX device        │          │      nonlinear product   │
                │                     │          │      FFT (downsample)    │
                └─────────────────────┘          │      k_at_z(z) via LUT   │
                                                 │    Dormand-Prince RK45   │
                                                 │    adaptive step control │
                                                 │                          │
                                                 │  a_out = IFFT(y_final)   │
                                                 └────────────┬─────────────┘
                                                              │
                                                     numpy array on CPU
```

### Key design decisions

**Why JAX instead of CuPy?**  CuPy provides a NumPy-compatible GPU API, but
each CuPy call dispatches a separate GPU kernel from Python.  The NEE solver
evaluates `fnl()` seven times per RK45 step, each involving ~18 operations.
At ~15 μs Python dispatch overhead per operation, the GPU sits idle waiting
for Python most of the time (measured: 11% GPU utilization with CuPy).
JAX's `jit` compiles the entire loop into a single GPU program, eliminating
dispatch overhead entirely.

**Poling lookup table.**  Arbitrary Python poling functions (QPM, chirped,
apodized, aperiodic) cannot be called inside JIT-compiled code.  Instead,
`_build_poling_table()` pre-samples the z-dependent coupling `k(z)` at high
resolution before compilation.  Inside the JIT loop, `k_at_z(z)` performs
O(1) nearest-neighbor lookup on the uniform grid — no interpolation
smoothing, preserving the sharp ±1 transitions of square-wave poling.

The sampling density is determined automatically from the poling pattern
(at least 200 samples per sign change, minimum 50,000 total).  For very
fine or aperiodic structures, pass `poling_samples=N` to override.

**Separability.**  The coupling `k(z)` is factored as `g(z) * k_shape` where
`k_shape` is the frequency-dependent part (computed once) and `g(z)` is the
scalar poling envelope (looked up per step).  This is valid for all practical
chi(2) waveguides where the nonlinear coefficient has a fixed spectral shape
modulated by a z-dependent poling pattern.

## Usage

```python
# Through the waveguide interface (recommended)
import jax.numpy as jnp
poling_jax = lambda z: jnp.sign(jnp.cos(z * 2*jnp.pi / pp))
out, steps = wg.propagate_NEE(pulse, v_ref=v_ref, backend='jax',
                              poling_fn_jax=poling_jax)

# Or call NEE directly for bulk crystal simulations
import snow.nlo_jax as nlo_jax
a_out, steps = nlo_jax.NEE(
    t=pulse.t, x=pulse.a, Omega=Omega, f0=pulse.f0,
    L=L, D=D, b0=beta_ref, b1_ref=beta_1_ref, k=k,
    poling_fn_jax=poling_jax
)
```

### JAX-native poling (recommended)

For best accuracy, pass the poling function as a JAX-compatible callable
using `jnp` operations.  This is evaluated directly inside the JIT loop,
giving bit-exact results matching the CPU solver for standard poling patterns.

```python
import jax.numpy as jnp

# Uniform QPM
poling_jax = lambda z: jnp.sign(jnp.cos(z * 2*jnp.pi / pp))

# Chirped QPM (period varies linearly with z)
chirp_rate = 0.5e-6  # period change per meter
poling_jax = lambda z: jnp.sign(jnp.cos(z * 2*jnp.pi / (pp + chirp_rate * z)))

# Apodized QPM (Gaussian-enveloped)
L = 4e-3
poling_jax = lambda z: (jnp.exp(-((z - L/2) / (L/4))**2)
                        * jnp.sign(jnp.cos(z * 2*jnp.pi / pp)))
```

Without `poling_fn_jax`, the solver falls back to a pre-sampled lookup table
(nearest-neighbor interpolation), which can introduce small discretization
errors that accumulate over long crystals.  The lookup table is still useful
for arbitrary poling patterns that can't be expressed in `jnp` ops (e.g.,
patterns loaded from a file).

### Pre-built poling table (for parameter sweeps)

When sweeping crystal length L with lookup-table mode, pre-build the table
once for the maximum L to avoid recompilation:

```python
ptable = nlo_jax.build_poling_table(k_func, L_max, 0, N)
for L in L_values:
    out, _ = wg.propagate_NEE(pulse, backend='jax', poling_table=ptable)
```

### Tolerances

`rtol` and `atol` are exposed as dynamic arguments (changing them does not
trigger recompilation):

```python
wg.propagate_NEE(pulse, backend='jax', rtol=1e-6, atol=1e-6)
```

The first call for each array size incurs a ~3-6 second JIT compilation cost.
Subsequent calls with the same `N` reuse the cached compiled program.

### Requirements

```
pip install "jax[cuda12]"
```

JAX automatically enables float64 precision (`jax_enable_x64`).

### Memory

JAX normally pre-allocates ~75% of GPU memory at startup.  `snow.nlo_jax`
overrides this default to **grow-on-demand** (`XLA_PYTHON_CLIENT_PREALLOCATE=false`)
so that multiple Jupyter notebook kernels can share a single GPU without
fighting over VRAM.  This has negligible performance impact since JAX still
pools freed allocations.

To restore JAX's default pre-allocation (e.g. for dedicated benchmarking),
set the environment variable before importing:

```python
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
# or: os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'

from snow import nlo_jax  # now pre-allocates
```

## Validation

The JAX solver is validated against the original SciPy-based CPU solver
(`scipy.integrate.RK45`) using a ladder of increasingly complex physics:

| # | Test                         | Field Correlation |
|---|------------------------------|-------------------|
| 1 | Linear, lossless, 1mm        | 1.000000          |
| 2 | Linear, lossless, 4mm        | 1.000000          |
| 3 | Linear, 0.5 dB/cm loss, 4mm  | 1.000000          |
| 4 | Weak nonlinear, 1mm          | 0.999876          |
| 5 | Full SHG, 4mm                | 0.999947          |
| 6 | Chirped QPM, 4mm             | 0.999592          |
| 7 | Apodized QPM, 4mm            | 0.999915          |
| 8 | Uniform QPM, 10mm            | 0.999279          |
| 9 | SHG + 0.3 dB/cm loss, 4mm    | 0.999950          |

"Field correlation" is the normalized overlap
`|<a_ref|a_jax>| / sqrt(<a_ref|a_ref> <a_jax|a_jax>)`.
All cases exceed 0.999.

### Running validation

**Fixed ladder** (the table above):
```bash
python test/cross_validate.py --reference /path/to/nlo_original.py
```

**Random fuzz testing** (run until Ctrl-C):
```bash
python test/cross_validate.py --soak --reference /path/to/nlo_original.py
```

This randomly samples from physically reasonable parameter ranges:
- Grid size N: 2^9 to 2^11
- Crystal length: 0.5 - 15 mm
- Nonlinear coefficient X0: 0 - 5e-12 m/V
- Loss: 0 - 1 dB/cm
- Poling period: 3 - 8 μm
- Pulse width: 50 - 300 fs
- Average power: 0.1 - 50 μW
- Poling type: uniform, chirped, or apodized (random)

Each iteration runs both solvers and reports the field correlation.

**Bounded fuzz** (e.g. 100 iterations):
```bash
python test/cross_validate.py --soak -n 100 --reference /path/to/nlo_original.py
```

### Known limitations

At high pump powers (>~50 μW average, corresponding to significant pump
depletion), the two adaptive solvers can diverge because the problem becomes
sensitive to the exact step sequence.  Both solvers remain individually valid
(conserve energy, produce physical spectra), but their field correlation
drops below 0.99.  This is inherent to adaptive ODE solvers on sensitive
problems, not a bug in either implementation.

## Performance

Benchmarked on RTX 4090 vs dual Xeon E5-2699 v3 (64 cores, OpenBLAS):

| N     | L    | JAX GPU | CPU (SciPy) | Speedup |
|-------|------|---------|-------------|---------|
| 1024  | 4mm  | 6.75s   | 57.6s       | 8.5x    |
| 4096  | 4mm  | 3.91s   | 242.2s      | 62x     |
| 16384 | 4mm  | 3.95s   | 949.0s      | 240x    |

JAX execution time is nearly independent of N (the GPU absorbs larger FFTs
without significant slowdown), while the CPU solver scales poorly.
Compilation adds ~3-6 seconds on first call per array shape.

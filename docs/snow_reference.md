# SNOW Technical Reference

A practical guide to the internals of the `snow` nonlinear photonics
simulator.  Covers conventions, normalization, parameter meanings,
and recipes for setting up simulations beyond what the tutorials show.

## Overview

SNOW solves the **Nonlinear Envelope Equation (NEE)** for broadband
chi(2) pulse propagation.  The NEE uses a single complex envelope `a(t)`
that contains all spectral components (pump, signal, idler, harmonics)
simultaneously.  All three-wave mixing processes (SHG, DFG, OPA, OPO)
emerge naturally from the same equation — you just set up the right
input conditions and let the NEE handle the physics.

**Reference:** M. Conforti, F. Baronio, and C. De Angelis, "Nonlinear
envelope equation for broadband optical pulses in quadratic media,"
Phys. Rev. A 81, 053841 (2010).

---

## Field normalization

**|a(t)|² = power in Watts.**

This is a power normalization, not intensity.  The field envelope `a(t)`
has units of sqrt(Watts).  Energy is computed as:

    E = integral(|a(t)|² dt)    [Joules]

This is confirmed by `pulses.energy_td()` and the Gaussian/sech peak
power formulas:

    Gaussian: Ppeak = 0.94 * Energy / FWHM
    Sech:     Ppeak = 0.88 * Energy / FWHM

For waveguide simulations, this convention is natural because the mode
area is folded into the coupling coefficient X0.  For bulk crystal
simulations, you need to account for the beam area — see the X0 section.

---

## Time and frequency grids

The simulation grid is defined by three quantities:

- **Bandwidth BW** = c/λ_start - c/λ_stop  (wavelength range to simulate)
- **N** = number of grid points (usually a power of 2)
- **dt** = 1/BW  (time resolution, set by bandwidth)

From these:

    T_window = N * dt = N / BW    (total time window)
    df = BW / N = 1/T_window      (frequency resolution)

The frequency grid is at **baseband** (centered at zero).  Physical
(absolute) frequencies are obtained by adding a reference frequency:

    f_abs = fftfreq(N, dt) + f_ref

**Choosing f_ref:**  Any value works for computation.  Convention is to
set it near the center of the spectral range of interest.  For SHG at
2um → 1um, use `f_ref = (c/800nm + c/3um)/2`.  For OPA with pump at
1um and signal at 1.5um, `f_ref = c/1.5um` works well.

**Choosing N:**  Larger N gives finer frequency resolution and wider
time window, but costs proportionally more computation.  Practical
guidelines:

| Application | N | Notes |
|---|---|---|
| SHG, Tutorial 5 | 2^10 (1024) | 800nm-3um, 100fs pulses |
| Broadband OPO | 2^10-2^12 | 800nm-3um, multi-roundtrip |
| Non-degenerate OPA | 2^14 (16384) | 900nm-4.2um, 10ps pulses |
| Supercontinuum | 2^14-2^16 | Very wide bandwidth |

---

## The pulse class

`pulses.pulse(t, x, wavelength, frep, domain='Time')`

- `t`: time array
- `x`: field envelope (complex array, |x|² = Watts)
- `wavelength`: reference wavelength (meters) — sets f0 = c/wavelength
- `frep`: pulse repetition rate (Hz) — used for average power calculations
- `domain`: 'Time' or 'Freq'

### Key attributes

| Attribute | Meaning |
|---|---|
| `pulse.a` | Time-domain envelope (complex) |
| `pulse.A` | Frequency-domain envelope (FFT of a) |
| `pulse.t` | Time array |
| `pulse.f` | Relative frequency array (baseband) |
| `pulse.f_abs` | Absolute frequency array (f + f0) |
| `pulse.f0` | Center frequency (c/wavelength) |
| `pulse.wl` | Wavelength array (c/f_abs) |
| `pulse.Omega` | Angular frequency array (2*pi*f) |
| `pulse.frep` | Repetition rate |

### fftshift convention

The pulse class applies `fftshift` when converting between time and
frequency domains.  This means `pulse.a` is centered (t=0 in the middle),
while the FFT convention puts t=0 at the start.

**Important:** `pulse.apply_filter()` returns a new pulse whose time-domain
field may be wrapped to the window boundary due to this convention.  For
reliable temporal profiles of filtered pulses, use direct FFT filtering:

```python
A = np.fft.fft(pulse.a)
mask = np.abs(f_abs - f_center) < bandwidth/2
a_filtered = np.fft.ifft(A * mask)
plt.plot(t, np.abs(a_filtered)**2)  # correct temporal profile
```

### Convenience pulse generators

```python
pulses.gaussian_pulse(t, FWHM, f_ref, Energy=, Ppeak=, Pavg=, f0=, Npwr_dB=, frep=)
pulses.sech_pulse(t, FWHM, f_ref, Energy=, Ppeak=, Pavg=, f0=, Npwr_dB=, frep=)
```

Specify energy via exactly one of `Energy`, `Ppeak`, or `Pavg`.

`Npwr_dB`: noise power relative to pulse peak (dB below).  Set to 200
for essentially noiseless pulses.  The noise is white Gaussian in both
quadratures.

`f0`: carrier frequency offset from f_ref.  The pulse is centered at f0
in the frequency domain.  Example: for a pump at 1um on a grid with
f_ref = c/1.5um, use `f0 = c/1um`.

---

## Waveguide class

```python
wg = waveguides.waveguide(w_top=1800e-9, h_thinfilm=700e-9, h_etch=350e-9,
                          tf_material='LN_MgO_e', box_material='SiO2',
                          clad_material='Air')
```

Models a ridge waveguide using the **effective index method** — a
semi-analytical approximation that solves 1D slab modes in each direction
and combines them.  Not a full-wave mode solver.

### Key methods

| Method | Returns |
|---|---|
| `wg.neff(wl)` | Effective index at wavelength(s) wl |
| `wg.beta1(wl)` | Group delay (1/vg) at wavelength wl |
| `wg.beta2(wl)` | GVD at wavelength wl |
| `wg.GVD(wl)` | Same as beta2 (alias) |

All wavelength arguments are in **meters** (not microns).

### Nonlinear setup

```python
wg.add_poling(poling_func)      # poling_func(z) returns +1 or -1
wg.set_nonlinear_coeffs(N=1, X0=1.1e-12)
wg.set_length(L)                # crystal length in meters
wg.set_loss(alpha)              # absorption coefficient in 1/m
```

- `poling_func(z)`: a Python function returning the poling sign at
  position z (meters).  For uniform QPM: `lambda z: np.sign(np.cos(z*2*pi/pp))`
- `N`: mode overlap parameter (typically 1 for single-mode waveguides)
- `X0`: nonlinear coupling coefficient (see dedicated section below)
- `alpha`: power absorption coefficient in 1/m.  Convert from dB/cm
  using `util.absorption_coeff(Alpha_dBcm)`.
- `gamma_eff`: effective Kerr nonlinearity parameter (1/W/m).  Pass as
  the third argument to `set_nonlinear_coeffs(N, X0, gamma_eff)`.
  Adds instantaneous SPM: the Kerr term `−i·γ·|A|²·A` in SNOW's
  dispersion convention (positive γ = self-focusing).  Default 0.
  Typical TFLN values: 100–400 /W/km = 0.1–0.4 /W/m.

### Running a simulation

```python
out_pulse, steps = wg.propagate_NEE(input_pulse, v_ref=v_ref,
                                     backend='scipy',  # or 'jax'
                                     verbose=True)
```

- `v_ref`: reference frame velocity (m/s).  The simulation runs in a
  co-moving frame at this velocity.  Choose it so the feature of interest
  stays near t=0.  Common choices:
  - `1/wg.beta1(lam_pump)` for pump-centered frame
  - `1/wg.beta1(lam_SH)` for SH-centered frame
  - `1/wg.beta1(lam_signal)` for signal-centered frame

- `backend`: `'scipy'` uses the original SciPy RK45 solver (CPU).
  `'jax'` uses the JAX JIT-compiled GPU solver.

- `z_save`: optional sorted array of z-positions (meters) at which to
  record the time-domain field during propagation.  When provided, the
  second return value is a 2-D complex array of shape
  `(len(z_save), NFFT)` containing field snapshots instead of the
  step-size array.  Uses the RK45 dense output for accurate
  interpolation at arbitrary z.  Currently supported only with
  `backend='scipy'`.

- Returns: `(output_pulse, step_list)` where step_list contains the
  adaptive step sizes used by the RK45 solver.  If `z_save` is
  provided, step_list is replaced by the snapshots array.

---

## The X0 coupling coefficient

This is the least documented and most confusing parameter in SNOW.

### For waveguides (Tutorials 5-7)

`X0 = 1.1e-12` is the value used in the TFLN waveguide tutorials.  It
represents the effective nonlinear coupling strength including the mode
overlap integral.  The coupling in the NEE is:

    k(z) = poling(z) * X0 * omega_abs / (4 * N_mode)

where `omega_abs` is the absolute angular frequency array and `N_mode`
is the mode overlap parameter (typically 1).

This value was presumably calibrated for the specific waveguide geometry
(1800nm top width, 700nm LN film, 350nm etch).  Changing the waveguide
geometry changes the mode overlap and thus X0, but SNOW does not compute
this automatically — you need to know X0 for your geometry.

### For bulk crystals (Tutorial 8)

For bulk crystal plane-wave propagation, X0 must account for the beam
area because the fields are power-normalized (|a|² = Watts) while the
nonlinear coupling involves intensity.

The first-principles formula `X0 = 4 * d_eff / (n * c)` gives the wrong
answer by a factor of ~20.  The correct value must be **calibrated
against the analytical small-signal OPA gain**:

    G = cosh²(g·L)
    g = sqrt(ω_s · ω_i · d_eff² · I_pump / (n_s · n_i · ε₀ · c³))

For PPLN (d_eff = 17.2 pm/V) with 100 um beam waist, X0 = 1.2e-14
matches the analytical gain within 1 dB.  The factor of 20 between
the formula and the calibrated value is not understood — it likely
relates to how the NEE's nonlinear product `a*(a·cos(φ) + 2·a*)`
maps to the standard coupled-mode parametric gain coefficient.

**Recipe for new geometries:**

1. Set up the simulation with a weak pump and seed
2. Compute the analytical g·L from physical parameters
3. Run the simulation, measure the signal gain
4. Adjust X0 until the simulation matches cosh²(g·L) for g·L < 2
5. Verify that pump depletion is < 5% at this operating point

---

## NEE solver details

### The nonlinear function

The core of the NEE is the function `fnl(z, y)` in `nlo.py`:

```
y_fast = y * exp(-j*D*z)           # remove dispersion phase
aup = upsample(y_fast)             # 4x zero-pad FFT for anti-aliasing
xup = aup * (cos(φ) + j*sin(φ))   # carrier modulation
f1up = aup * (xup + 2*conj(xup))  # chi(2) nonlinear product
F1 = downsample(FFT(f1up))        # back to original grid
return -j * k(z) * F1 * exp(j*D*z) # apply coupling, restore phase
```

The nonlinear product `a*(a·e^(jφ) + 2·a*)` contains both SHG (a²)
and DFG/OPA (|a|²·a) terms simultaneously.  The carrier phase φ
evolves with z according to the reference wavevector, providing
the phase-matching condition.

### Upsampling

The 4x upsampling before the nonlinear product prevents aliasing
from the frequency mixing.  For very broadband simulations where
3·f_max > 4·BW, 8x upsampling is used automatically (with a warning).

### Adaptive stepping

The RK45 solver uses the Dormand-Prince method with adaptive step
control.  Default tolerances: rtol = atol = 1e-4.

The number of steps depends on:
- Crystal length
- Nonlinear coupling strength (stronger = more steps)
- Poling period (the solver resolves the poling transitions)
- Spectral bandwidth (more bandwidth = more phase dynamics)

Typical step counts:
- 4mm TFLN SHG at 1uW: ~200 steps
- 20mm PPLN OPA at 5nJ: ~1200 steps
- 10mm TFLN OPO roundtrip at 500uW: ~400 steps

---

## Material models

`materials.refractive_index(material, wl, T=24.5)`

Wavelength units are auto-detected (meters, nm, or microns).

### Available materials

| Name | Material | Source |
|---|---|---|
| `'SiO2'` | Fused silica | Malitson 1965 |
| `'Sapphire'` | Al2O3 | Malitson |
| `'LN_MgO_e'` | 5% MgO:LN extraordinary | Zelmon-type |
| `'LN_MgO_o'` | 5% MgO:LN ordinary | Zelmon-type |
| `'LN_e'` | Undoped LN extraordinary | Standard |
| `'LN_o'` | Undoped LN ordinary | Standard |
| `'LN_MgO_e_T'` | 5% MgO:LN e, **temp-dependent** | Gayer 2008 |
| `'LN_MgO_o_T'` | 5% MgO:LN o, **temp-dependent** | Gayer 2008 |
| `'GaP'` | Gallium phosphide | Aspnes |
| `'LT_MgO_e'` | MgO:LiTaO3 extraordinary | — |
| `'LT_SLT'` | Stoichiometric LiTaO3 | Brunner 2003 |
| `'Air'` | n = 1 | — |

Temperature-dependent models (`_T` suffix) are required for OPA
temperature tuning simulations.  Valid range: approximately 0.5-5 um.

---

## Solver backends

### SciPy (CPU)

`backend='scipy'` — the original solver.  Uses `scipy.integrate.RK45`.

- Battle-tested, deterministic
- Performance scales with N and crystal length
- Uses all available CPU cores (via BLAS threading)
- Best for: small grids (N ≤ 2^12), quick single-pass calculations

### JAX (GPU)

`backend='jax'` — JIT-compiled GPU solver.

- Faithful port of SciPy's RK45 (FSAL, identical step control)
- ~3-6s compilation on first call per grid size
- Bit-identical to SciPy for short crystals; may diverge at long
  crystals due to GPU vs CPU floating-point arithmetic
- Best for: large grids (N ≥ 2^12), parameter sweeps, OPO roundtrips

**JAX-native poling:**  For best accuracy, pass the poling function
as a JAX-compatible callable:

```python
import jax.numpy as jnp
poling_jax = lambda z: jnp.sign(jnp.cos(z * 2*jnp.pi / pp))
wg.propagate_NEE(pulse, backend='jax', poling_fn_jax=poling_jax)
```

This evaluates the poling directly inside the JIT loop, avoiding
lookup table discretization errors.

**Pre-built poling table:**  For parameter sweeps where L changes,
pre-build the table once for the maximum L:

```python
from snow import nlo_jax
ptable = nlo_jax.build_poling_table(k_func, L_max, 0, N)
wg.propagate_NEE(pulse, backend='jax', poling_table=ptable)
```

**Memory:** JAX defaults to grow-on-demand GPU allocation (set by
`snow.nlo_jax` on import).  Multiple notebook kernels can share one GPU.

---

## Recipes

### SHG (Tutorial 5 pattern)

```python
pump = pulses.sech_pulse(t, 100*fs, f_ref=f_ref, f0=c/(2*um),
                         Pavg=1e-6, Npwr_dB=200, frep=250e6)
wg.add_poling(lambda z: np.sign(np.cos(z*2*pi/pp)))
wg.set_nonlinear_coeffs(N=1, X0=1.1e-12)
out, _ = wg.propagate_NEE(pump, v_ref=1/wg.beta1(1*um))
```

### OPO (Tutorial 7 pattern)

```python
recycled = pulses.noise(t, 0)
for rt in range(n_roundtrips):
    pump = pulses.sech_pulse(t, 100*fs, f_ref=f_ref, f0=c/(1*um),
                             Pavg=500e-6, Npwr_dB=200, frep=250e6)
    sig = pulses.pulse(t, recycled, c/f_ref, frep)
    inp = pump + sig
    out, _ = wg.propagate_NEE(inp, v_ref=v_ref, Qnoise=True)
    sig_out = out.apply_filter(f0_signal, signal_bw)
    recycled = sig_out.a * np.sqrt(feedback_frac)
```

### z-resolved field evolution

To record the field at intermediate propagation points (e.g. for
(z, t) color maps of soliton dynamics), pass `z_save`:

```python
z_positions = np.linspace(0.5*mm, L, 100)
out, snapshots = wg.propagate_NEE(pulse, v_ref=v_ref, z_save=z_positions)

# snapshots is complex, shape (100, NFFT) — time-domain field at each z
plt.pcolormesh(t/fs, z_positions/mm, np.abs(snapshots)**2)
plt.xlabel('Time (fs)'); plt.ylabel('z (mm)')
```

Each snapshot is the time-domain field `a(z, t)` where `|a|² = Watts`.
The SciPy backend uses RK45 dense output for accurate interpolation
at arbitrary z.  The JAX backend clamps steps to land exactly on
`z_save` positions.  Both backends are supported.

### Bulk crystal OPA (Tutorial 8 pattern)

For bulk crystals without waveguide confinement, call `nlo.NEE`
directly with custom dispersion and coupling:

```python
# Compute dispersion from Sellmeier model
n_grid = [materials.refractive_index('LN_MgO_e_T', wl, T=150)
          for wl in wavelengths]
beta = 2*pi*f_abs * n_grid / c
D = beta - beta_ref - Omega/v_ref

# Coupling (calibrated X0)
X0 = 1.2e-14  # for 100um beam waist in PPLN
k_func = lambda z: np.sign(np.cos(z*2*pi/pp)) * X0 * omega_abs / 4

# Propagate
a_out, steps = nlo_jax.NEE(t=t, x=input.a, Omega=Omega, f0=f_ref,
                           L=L, D=D, b0=beta_ref, b1_ref=1/v_ref,
                           k=k_func, poling_fn_jax=poling_jax)
```

---

## Known issues and limitations

1. **pulse.apply_filter fftshift wrapping:** Filtered pulses may have
   their temporal profile wrapped to the window boundary.  Use direct
   FFT filtering for temporal plots (see pulse class section).

2. **X0 normalization for bulk crystals:** The first-principles formula
   gives the wrong X0 by a factor of ~20.  Must be calibrated against
   analytical gain.  The derivation is an open question.

3. **High-gain parametric generator regime:** At g·L > 3, the pump's
   own spectral tail (from the finite-duration pulse) provides enough
   seed to saturate the parametric process.  The output becomes
   independent of the external seed.  This is correct physics, not a
   bug — but it means you can't study OPA gain at high pump energies
   without reducing the crystal length or beam size.

4. **No chi(3):** The NEE only includes chi(2) nonlinearity.  Kerr
   effect, Raman scattering, and self-phase modulation are not modeled.

5. **No diffraction:** The simulation is 1D (propagation axis only).
   Beam divergence, focusing, and spatial mode effects are not included.

6. **Waveguide effective index method:** The semi-analytical mode
   solver is approximate.  For precise mode properties, use the
   Lumerical interface (`lumerical.py`) for full-wave simulation
   and import the results.

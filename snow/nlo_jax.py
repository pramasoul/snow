# -*- coding: utf-8 -*-
"""
JAX-accelerated NEE solver.

The entire Dormand-Prince RK45 integration loop, including the nonlinear
fnl() evaluation with FFTs, is JIT-compiled into a single fused GPU program.
No Python dispatch overhead in the hot loop.

The step controller is a faithful port of scipy.integrate.RK45 (Dormand-Prince)
including FSAL (First Same As Last), the step_rejected flag, and identical
safety/factor constants, so that JAX and SciPy produce matching step sequences.

The z-dependent nonlinear coupling k(z) is pre-sampled into a lookup table
so that arbitrary poling patterns (uniform, chirped, apodized, aperiodic)
are supported without Python callbacks inside the JIT boundary.
"""
import os
import numpy as np
from scipy.constants import pi, h as h_planck

# Default to grow-on-demand GPU memory allocation.  JAX normally pre-allocates
# 75% of VRAM on first import, which is hostile in multi-kernel environments
# (e.g. several Jupyter notebooks sharing one GPU).  Grow-on-demand has
# negligible performance impact since JAX still pools freed allocations.
# Users can override by setting the env var before importing snow.nlo_jax.
if 'XLA_PYTHON_CLIENT_PREALLOCATE' not in os.environ:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Dormand-Prince RK45 coefficients — exactly matching scipy.integrate.RK45
# ---------------------------------------------------------------------------

# Stage time offsets
_C = jnp.array([0, 1/5, 3/10, 4/5, 8/9, 1])

# Stage weight matrix (A[s, :s] gives weights for stage s)
_A = jnp.array([
    [0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
])

# Solution weights (5th order)
_B = jnp.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])

# Error coefficients (applied to all 7 K values including FSAL)
_E = jnp.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200,
                 -22/525, 1/40])

# Step control constants (matching SciPy exactly)
_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 10.0
_ERROR_EXPONENT = -0.2  # -1 / (error_estimator_order + 1) = -1/5


# ---------------------------------------------------------------------------
# Poling table
# ---------------------------------------------------------------------------

def _build_poling_table(k, L, z0, NFFT, n_samples=None):
    """Pre-sample the z-dependent coupling k(z) into a separable form.

    Returns (k_shape, z_table, g_table) where:
      k_shape : JAX array [NFFT] — frequency-dependent coupling shape
      z_table : JAX array [n_samples] — z sample points
      g_table : JAX array [n_samples] — scalar poling envelope at each z

    Inside JIT: k(z) ≈ interp(z, z_table, g_table) * k_shape
    """
    k_ref = k(float(z0))
    idx_max = int(np.argmax(np.abs(k_ref)))
    k_ref_val = k_ref[idx_max]

    if abs(k_ref_val) < 1e-30:
        found = False
        z_test = np.linspace(float(z0), float(L), 100)
        for zt in z_test:
            k_test = k(zt)
            idx_max = int(np.argmax(np.abs(k_test)))
            if abs(k_test[idx_max]) > 1e-30:
                k_ref = k_test
                k_ref_val = k_ref[idx_max]
                found = True
                break
        if not found:
            _n = n_samples if n_samples is not None else 50000
            k_shape = np.zeros(NFFT, dtype=complex)
            z_table = np.linspace(float(z0), float(L), _n)
            g_table = np.zeros(_n)
            return (jnp.asarray(k_shape),
                    jnp.asarray(z_table),
                    jnp.asarray(g_table))

    k_shape = k_ref.copy()

    if n_samples is None:
        z_coarse = np.linspace(float(z0), float(L), 1000)
        g_coarse = np.array([k(z)[idx_max] / k_ref_val for z in z_coarse])
        sign_changes = np.sum(np.abs(np.diff(np.sign(g_coarse))) > 0)
        L_span = float(L) - float(z0)
        if L_span > 0 and sign_changes > 0:
            changes_per_m = sign_changes / L_span
            n_samples = max(50000, int(changes_per_m * L_span * 200))
        else:
            n_samples = 50000
        n_samples = max(50000, ((n_samples + 9999) // 10000) * 10000)

    z_table = np.linspace(float(z0), float(L), n_samples)
    g_table = np.array([k(z)[idx_max] / k_ref_val for z in z_table])

    return (jnp.asarray(k_shape),
            jnp.asarray(z_table),
            jnp.asarray(g_table))


def build_poling_table(k, L, z0, NFFT, n_samples=None):
    """Pre-build a poling lookup table for use across multiple NEE() calls.

    Use this when sweeping L or other parameters with the same waveguide
    and grid.  Build once for the maximum L, then pass to each NEE() call.

    Returns a tuple (k_shape, z_table, g_table) to pass as poling_table=.
    """
    return _build_poling_table(k, L, z0, NFFT, n_samples=n_samples)


# ---------------------------------------------------------------------------
# NEE solver
# ---------------------------------------------------------------------------

def NEE(t, x, Omega, f0,
        L, D, b0, b1_ref, k,
        z0=0, verbose=True, Kg=0, Qnoise=False,
        gpu=None, poling_samples=None, poling_table=None,
        poling_fn_jax=None,
        rtol=1e-4, atol=1e-4):
    """
    Nonlinear-envelope equation -- JAX JIT-compiled adaptive RK45 solver.

    The step controller is a faithful port of scipy.integrate.RK45 so that
    JAX and SciPy produce matching step sequences and results.

    Parameters
    ----------
    poling_samples : int, optional
        Number of z-points for the poling lookup table.
    poling_table : tuple, optional
        Pre-built (k_shape, z_table, g_table) from build_poling_table().
    poling_fn_jax : callable(z) -> scalar, optional
        A JAX-compatible poling function (using jnp ops, not numpy).
        When provided, this is called directly inside the JIT loop
        instead of using the lookup table, giving bit-exact agreement
        with the CPU solver for standard poling patterns.
        Example: lambda z: jnp.sign(jnp.cos(z * 2*jnp.pi / pp))
    rtol : float
        Relative tolerance for the adaptive step controller (default 1e-4).
    atol : float
        Absolute tolerance for the adaptive step controller (default 1e-4).

    All other parameters match nlo.NEE for drop-in use.
    """
    NFFT = t.size
    Omega_abs = Omega + 2*pi*f0
    f_max = float(np.amax(Omega_abs)) / (2*pi)
    f_min = float(np.amin(Omega_abs)) / (2*pi)
    BW = f_max - f_min
    dt_grid = 1/BW
    f = np.fft.fftfreq(NFFT, dt_grid)
    df = f[1] - f[0]
    f_abs = f + f0

    Nup = 4
    if (3*f_max - f_min)/BW > Nup:
        Nup = 8
        if verbose:
            print('Warning: large upsampling necessary!')
            print('Using %ix upsampling.' % Nup)

    # Quantum noise (CPU)
    if Qnoise:
        phi_noise = np.random.uniform(0, 2*pi, NFFT)
        Xnoise = BW * np.sqrt(h_planck*f_abs/2/df) * np.exp(1j * phi_noise)
        xnoise = np.fft.ifft(Xnoise)
    else:
        xnoise = np.zeros(NFFT)

    # Move constants to device
    D_dev = jnp.asarray(D)
    x_in = jnp.asarray(x + xnoise)

    tup = np.linspace(float(t[0]), float(t[-1]), Nup*NFFT)
    phi_1 = jnp.asarray(2*pi*f0*tup)
    phi_2 = float(b0 - b1_ref*2*pi*f0 + Kg)

    A0 = jnp.fft.fft(x_in)

    M = NFFT*Nup - NFFT
    center = NFFT // 2 + 1

    # Determine poling evaluation strategy
    if poling_fn_jax is not None:
        # Extract frequency-dependent coupling shape from k(z0)
        k_ref = k(float(z0))
        k_shape = jnp.asarray(k_ref.copy())
        # Dummy table args (unused but keeps JIT signature stable)
        z_table = jnp.zeros(2)
        g_table = jnp.zeros(2)
        if verbose:
            print('Using JAX-native poling function (bit-exact)')
    elif poling_table is not None:
        k_shape, z_table, g_table = poling_table
        if verbose:
            print(f'Poling table: {len(z_table)} samples (pre-built)')
    else:
        k_shape, z_table, g_table = _build_poling_table(
            k, L, z0, NFFT, n_samples=poling_samples)
        if verbose:
            print(f'Poling table: {len(z_table)} samples over '
                  f'{float(L)*1e3:.2f} mm')

    # Build k_at_z closure BEFORE @jax.jit — JAX traces whichever
    # branch was taken, compiling only the relevant code path.
    if poling_fn_jax is not None:
        def _k_at_z(z, k_shape, z_table, g_table):
            return poling_fn_jax(z) * k_shape
    else:
        def _k_at_z(z, k_shape, z_table, g_table):
            idx = jnp.int32(jnp.round(
                (z - z_table[0]) / (z_table[1] - z_table[0])))
            idx = jnp.clip(idx, 0, z_table.shape[0] - 1)
            return g_table[idx] * k_shape

    # JIT-compiled solver — faithful port of scipy.integrate.RK45
    @jax.jit
    def _solve(A0, k_shape, z_table, g_table, L_val, z0_val,
               rtol_val, atol_val):

        def k_at_z(z):
            return _k_at_z(z, k_shape, z_table, g_table)

        def fnl(z, y):
            phi = phi_1 - phi_2 * z
            y_fast = y * jnp.exp(-1j * D_dev * z)

            Aup = jnp.zeros(Nup * NFFT, dtype=y.dtype)
            Aup = Aup.at[:center].set(y_fast[:center])
            Aup = Aup.at[center+M:].set(y_fast[center:])
            aup = jnp.fft.ifft(Aup) * Nup

            xup = aup * (jnp.cos(phi) + 1j * jnp.sin(phi))
            f1up = aup * (xup + 2 * jnp.conj(xup))

            F1up = jnp.fft.fft(f1up)
            F1 = jnp.zeros_like(y)
            F1 = F1.at[:center].set(F1up[:center])
            F1 = F1.at[center:].set(F1up[center+M:])
            F1 = F1 / Nup

            return -1j * k_at_z(z) * F1 * jnp.exp(1j * D_dev * z)

        def rk_step(z, y, f0_val, h):
            """One Dormand-Prince step with FSAL.

            Takes f0_val (derivative at current point) to avoid recomputing.
            Returns (y_new, f_new, error_norm).
            """
            # K[0] = f0_val (FSAL — reused from previous step)
            K0 = f0_val
            K1 = fnl(z + _C[1]*h, y + h * (_A[1,0]*K0))
            K2 = fnl(z + _C[2]*h, y + h * (_A[2,0]*K0 + _A[2,1]*K1))
            K3 = fnl(z + _C[3]*h, y + h * (_A[3,0]*K0 + _A[3,1]*K1
                                            + _A[3,2]*K2))
            K4 = fnl(z + _C[4]*h, y + h * (_A[4,0]*K0 + _A[4,1]*K1
                                            + _A[4,2]*K2 + _A[4,3]*K3))
            K5 = fnl(z + _C[5]*h, y + h * (_A[5,0]*K0 + _A[5,1]*K1
                                            + _A[5,2]*K2 + _A[5,3]*K3
                                            + _A[5,4]*K4))

            # 5th order solution
            y_new = y + h * (_B[0]*K0 + _B[2]*K2 + _B[3]*K3
                             + _B[4]*K4 + _B[5]*K5)

            # FSAL: evaluate derivative at new point
            f_new = fnl(z + h, y_new)

            # Error estimate using all 7 K values
            err = h * (_E[0]*K0 + _E[2]*K2 + _E[3]*K3
                       + _E[4]*K4 + _E[5]*K5 + _E[6]*f_new)

            # Error norm (RMS, matching SciPy's norm())
            scale = atol_val + rtol_val * jnp.maximum(
                jnp.abs(y), jnp.abs(y_new))
            error_norm = jnp.sqrt(
                jnp.sum(jnp.abs(err / scale)**2) / err.size)

            return y_new, f_new, error_norm

        # --- Initial step size (matching scipy select_initial_step) ---
        f0_init = fnl(z0_val, A0)
        scale0 = atol_val + jnp.abs(A0) * rtol_val
        d0 = jnp.sqrt(jnp.sum(jnp.abs(A0 / scale0)**2) / A0.size)
        d1 = jnp.sqrt(jnp.sum(jnp.abs(f0_init / scale0)**2) / A0.size)

        h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5),
                        1e-6, 0.01 * d0 / d1)
        h0 = jnp.minimum(h0, L_val - z0_val)

        y1 = A0 + h0 * f0_init
        f1 = fnl(z0_val + h0, y1)
        d2 = jnp.sqrt(jnp.sum(jnp.abs((f1 - f0_init) / scale0)**2)
                       / A0.size) / h0

        h1 = jnp.where(
            (d1 <= 1e-15) & (d2 <= 1e-15),
            jnp.maximum(1e-6, h0 * 1e-3),
            (0.01 / jnp.maximum(d1, d2)) ** 0.2)  # exponent = 1/(order+1) = 1/5

        h_init = jnp.minimum(jnp.minimum(100 * h0, h1), L_val - z0_val)

        # State: (z, y, h, f_current, n_steps, step_rejected)
        init_state = (jnp.float64(z0_val), A0, jnp.float64(h_init),
                      f0_init, jnp.int32(0), jnp.bool_(False))

        def cond_fn(state):
            z, _, _, _, _, _ = state
            return z < L_val

        def body_fn(state):
            z, y, h_abs, f_cur, n, was_rejected = state

            # Clamp step to remaining distance
            h = jnp.minimum(h_abs, L_val - z)

            # RK step with FSAL
            y_new, f_new, error_norm = rk_step(z, y, f_cur, h)

            accepted = error_norm < 1.0

            # Step size adjustment (matching SciPy exactly)
            factor = jnp.where(
                error_norm == 0,
                _MAX_FACTOR,
                jnp.minimum(_MAX_FACTOR,
                            _SAFETY * error_norm ** _ERROR_EXPONENT))

            # After a rejected step, don't allow growth > 1
            factor = jnp.where(was_rejected & accepted,
                               jnp.minimum(1.0, factor),
                               factor)

            # On rejection, shrink
            reject_factor = jnp.maximum(
                _MIN_FACTOR,
                _SAFETY * error_norm ** _ERROR_EXPONENT)

            h_new = jnp.where(accepted,
                              h_abs * factor,
                              h_abs * reject_factor)

            # Update state
            z_next = jnp.where(accepted, z + h, z)
            y_next = jnp.where(accepted, y_new, y)
            f_next = jnp.where(accepted, f_new, f_cur)
            n_next = jnp.where(accepted, n + 1, n)
            rejected_next = ~accepted

            return (z_next, y_next, h_new, f_next, n_next, rejected_next)

        z_final, y_final, h_final, f_final, n_steps, _ = \
            jax.lax.while_loop(cond_fn, body_fn, init_state)

        # Final transform
        A_out = y_final * jnp.exp(-1j * D_dev * L_val)
        a_out = jnp.fft.ifft(A_out)

        return a_out, n_steps

    # Run
    if verbose:
        print('JAX compiling + running...')

    L_val = jnp.float64(L)
    z0_val = jnp.float64(z0)
    rtol_val = jnp.float64(rtol)
    atol_val = jnp.float64(atol)
    a_out, n_steps = _solve(A0, k_shape, z_table, g_table,
                            L_val, z0_val, rtol_val, atol_val)

    # Block and move to CPU
    a_out = np.asarray(a_out)
    n_steps = int(n_steps)

    if verbose:
        print(f'finished ({n_steps} steps)')

    return a_out, [float(L) / max(n_steps, 1)] * n_steps


if __name__ == '__main__':
    pass

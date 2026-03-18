# -*- coding: utf-8 -*-
"""
JAX-accelerated NEE solver.

The entire Dormand-Prince RK45 integration loop, including the nonlinear
fnl() evaluation with FFTs, is JIT-compiled into a single fused GPU program.
No Python dispatch overhead in the hot loop.

The z-dependent nonlinear coupling k(z) is pre-sampled into a lookup table
so that arbitrary poling patterns (uniform, chirped, apodized, aperiodic)
are supported without Python callbacks inside the JIT boundary.
"""
import numpy as np
from scipy.constants import pi, h as h_planck

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# Dormand-Prince coefficients
_C = jnp.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
_D = jnp.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
_E = jnp.array([71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40])


def _build_poling_table(k, L, z0, NFFT, n_samples=None):
    """Pre-sample the z-dependent coupling k(z) into a separable form.

    Returns (k_shape, z_table, g_table) where:
      k_shape : JAX array [NFFT] — frequency-dependent coupling shape
      z_table : JAX array [n_samples] — z sample points
      g_table : JAX array [n_samples] — scalar poling envelope at each z

    Inside JIT: k(z) ≈ interp(z, z_table, g_table) * k_shape
    """
    # Reference evaluation at z=0
    k_ref = k(float(z0))
    idx_max = int(np.argmax(np.abs(k_ref)))
    k_ref_val = k_ref[idx_max]

    if abs(k_ref_val) < 1e-30:
        # Zero coupling — find a non-zero reference point
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
            # Truly zero nonlinearity — return zero coupling
            k_shape = np.zeros(NFFT, dtype=complex)
            z_table = np.array([float(z0), float(L)])
            g_table = np.zeros(2)
            return (jnp.asarray(k_shape),
                    jnp.asarray(z_table),
                    jnp.asarray(g_table))

    k_shape = k_ref.copy()  # frequency-dependent coupling (unnormalized)

    # Determine sampling density from the coupling pattern.
    # Use enough points to capture rapid poling transitions.
    if n_samples is None:
        # Estimate poling period from zero-crossings
        z_coarse = np.linspace(float(z0), float(L), 1000)
        g_coarse = np.array([k(z)[idx_max] / k_ref_val for z in z_coarse])
        sign_changes = np.sum(np.abs(np.diff(np.sign(g_coarse))) > 0)
        # At least 200 samples per sign change, minimum 50000 total
        # Dense sampling is critical: we use nearest-neighbor lookup,
        # so each sample must accurately represent its neighborhood
        n_samples = max(50000, sign_changes * 200)

    # Sample the scalar poling envelope
    z_table = np.linspace(float(z0), float(L), n_samples)
    g_table = np.array([k(z)[idx_max] / k_ref_val for z in z_table])

    return (jnp.asarray(k_shape),
            jnp.asarray(z_table),
            jnp.asarray(g_table))


def NEE(t, x, Omega, f0,
        L, D, b0, b1_ref, k,
        z0=0, verbose=True, Kg=0, Qnoise=False,
        gpu=None, poling_samples=None):
    """
    Nonlinear-envelope equation -- JAX JIT-compiled adaptive RK45 solver.

    Parameters
    ----------
    poling_samples : int, optional
        Number of z-points for the poling lookup table.  If None, determined
        automatically from the poling pattern.  Increase for very fine or
        aperiodic structures.

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

    # Build poling lookup table (all Python, before JIT boundary)
    k_shape, z_table, g_table = _build_poling_table(
        k, L, z0, NFFT, n_samples=poling_samples)

    if verbose:
        print(f'Poling table: {len(z_table)} samples over '
              f'{float(L)*1e3:.2f} mm')

    # Precompute table spacing for O(1) nearest-neighbor lookup
    z_start = float(z_table[0])
    z_step = float(z_table[1] - z_table[0])
    n_table = len(z_table)

    # JIT-compiled solver — all constants are captured as closure variables
    @jax.jit
    def _solve(A0, k_shape, z_table, g_table):

        def k_at_z(z):
            """Nearest-neighbor lookup of poling envelope at position z.
            O(1) — no search, just index arithmetic on uniform grid."""
            idx = jnp.int32(jnp.round((z - z_start) / z_step))
            idx = jnp.clip(idx, 0, n_table - 1)
            g = g_table[idx]
            return g * k_shape

        def fnl(z, y):
            phi = phi_1 - phi_2 * z
            y_fast = y * jnp.exp(-1j * D_dev * z)

            # Upsample via zero-padding
            Aup = jnp.zeros(Nup * NFFT, dtype=y.dtype)
            Aup = Aup.at[:center].set(y_fast[:center])
            Aup = Aup.at[center+M:].set(y_fast[center:])
            aup = jnp.fft.ifft(Aup) * Nup

            # Nonlinear product
            xup = aup * (jnp.cos(phi) + 1j * jnp.sin(phi))
            f1up = aup * (xup + 2 * jnp.conj(xup))

            # Downsample
            F1up = jnp.fft.fft(f1up)
            F1 = jnp.zeros_like(y)
            F1 = F1.at[:center].set(F1up[:center])
            F1 = F1.at[center:].set(F1up[center+M:])
            F1 = F1 / Nup

            return -1j * k_at_z(z) * F1 * jnp.exp(1j * D_dev * z)

        # --- Adaptive RK45 via lax.while_loop ---
        rtol = 1e-4
        atol = 1e-4

        def rk45_step(z, y, h):
            """One Dormand-Prince step. Returns (y_new, err_norm)."""
            k0 = fnl(z, y)
            k1 = fnl(z + _C[1]*h, y + h*(1/5*k0))
            k2 = fnl(z + _C[2]*h, y + h*(3/40*k0 + 9/40*k1))
            k3 = fnl(z + _C[3]*h, y + h*(44/45*k0 - 56/15*k1 + 32/9*k2))
            k4 = fnl(z + _C[4]*h, y + h*(19372/6561*k0 - 25360/2187*k1
                      + 64448/6561*k2 - 212/729*k3))
            k5 = fnl(z + _C[5]*h, y + h*(9017/3168*k0 - 355/33*k1
                      + 46732/5247*k2 + 49/176*k3 - 5103/18656*k4))

            y_new = y + h * (35/384*k0 + 500/1113*k2 + 125/192*k3
                             - 2187/6784*k4 + 11/84*k5)
            k6 = fnl(z + h, y_new)

            err = h * (71/57600*k0 - 71/16695*k2 + 71/1920*k3
                       - 17253/339200*k4 + 22/525*k5 - 1/40*k6)
            scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
            err_norm = jnp.sqrt(jnp.mean(jnp.abs(err / scale)**2))

            return y_new, err_norm

        # Initial step size
        f0_eval = fnl(z0, A0)
        scale0 = atol + rtol * jnp.abs(A0)
        d0 = jnp.sqrt(jnp.mean(jnp.abs(A0 / scale0)**2))
        d1 = jnp.sqrt(jnp.mean(jnp.abs(f0_eval / scale0)**2))
        h0 = jnp.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)
        h0 = jnp.minimum(h0, L - z0)
        y1 = A0 + h0 * f0_eval
        f1_eval = fnl(z0 + h0, y1)
        d2 = jnp.sqrt(jnp.mean(
            jnp.abs((f1_eval - f0_eval) / scale0)**2)) / h0
        h1 = jnp.where(jnp.maximum(d1, d2) <= 1e-15,
                        jnp.maximum(1e-6, h0 * 1e-3),
                        (0.01 / jnp.maximum(d1, d2)) ** 0.2)
        h_init = jnp.minimum(jnp.minimum(100 * h0, h1), L - z0)

        # State: (z, y, h, n_steps)
        init_state = (jnp.float64(z0), A0,
                      jnp.float64(h_init), jnp.int32(0))

        def cond_fn(state):
            z, _, _, _ = state
            return z < L

        def body_fn(state):
            z, y, h, n = state
            h = jnp.minimum(h, L - z)

            y_new, err_norm = rk45_step(z, y, h)

            accept = err_norm <= 1.0
            z_next = jnp.where(accept, z + h, z)
            y_next = jnp.where(accept, y_new, y)
            n_next = jnp.where(accept, n + 1, n)

            factor = jnp.where(err_norm > 1e-15,
                               0.9 * err_norm ** (-0.2), 5.0)
            factor = jnp.clip(factor, 0.2, 5.0)
            h_next = h * factor

            return (z_next, y_next, h_next, n_next)

        z_final, y_final, h_final, n_steps = jax.lax.while_loop(
            cond_fn, body_fn, init_state)

        # Final transform
        A_out = y_final * jnp.exp(-1j * D_dev * L)
        a_out = jnp.fft.ifft(A_out)

        return a_out, n_steps

    # Run
    if verbose:
        print('JAX compiling + running...')

    a_out, n_steps = _solve(A0, k_shape, z_table, g_table)

    # Block and move to CPU
    a_out = np.asarray(a_out)
    n_steps = int(n_steps)

    if verbose:
        print(f'finished ({n_steps} steps)')

    return a_out, [float(L) / max(n_steps, 1)] * n_steps


if __name__ == '__main__':
    pass

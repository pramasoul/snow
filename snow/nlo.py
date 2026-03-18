# -*- coding: utf-8 -*-
"""
@author: Luis Ledezma
GPU-ready refactor: custom RK45 replaces scipy.integrate.RK45 so the entire
integration loop stays on whichever array backend (numpy or cupy) the input
uses.  Falls back to numpy transparently when cupy is not available.
"""
import numpy as np
from scipy.constants import pi, h as h_planck

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


def to_cpu(x):
    """Ensure *x* is a numpy array on the host."""
    if _HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Dormand-Prince RK45 coefficients
# ---------------------------------------------------------------------------
_C = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
_B = [
    [],
    [1/5],
    [3/40, 9/40],
    [44/45, -56/15, 32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
]
_D = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
_E = [71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40]


# ---------------------------------------------------------------------------
# NEE solver
# ---------------------------------------------------------------------------

def NEE(t, x, Omega, f0,
        L, D, b0, b1_ref, k,
        z0=0, verbose=True, Kg=0, Qnoise=False,
        gpu=False):
    """
    Nonlinear-envelope equation -- adaptive RK45 solver.

    Set gpu=True to run on CUDA (requires cupy).  When gpu=False (default),
    behaviour is identical to the original numpy implementation.
    """
    use_gpu = gpu and _HAS_CUPY
    xp = cp if use_gpu else np

    # ----- grid setup (CPU) -----
    NFFT = t.size
    Omega_abs = Omega + 2*pi*f0
    f_max = float(np.amax(Omega_abs)) / (2*pi)
    f_min = float(np.amin(Omega_abs)) / (2*pi)
    BW = f_max - f_min
    dt = 1/BW
    f = np.fft.fftfreq(NFFT, dt)
    df = f[1] - f[0]
    f_abs = f + f0

    Nup = 4
    if (3*f_max - f_min)/BW > Nup:
        Nup = 8
        if verbose:
            print('Warning: large upsampling necessary!')
            print('Using %ix upsampling.' % Nup)

    # ----- quantum noise (CPU) -----
    if Qnoise:
        phi_noise = np.random.uniform(0, 2*pi, NFFT)
        Xnoise = BW * np.sqrt(h_planck*f_abs/2/df) * np.exp(1j * phi_noise)
        xnoise = np.fft.ifft(Xnoise)
    else:
        xnoise = np.zeros(NFFT)

    # ----- move everything to device once -----
    x_dev = xp.asarray(x + xnoise)
    D_dev = xp.asarray(D)

    tup = np.linspace(float(t[0]), float(t[-1]), Nup*NFFT)
    phi_1 = xp.asarray(2*pi*f0*tup)
    phi_2 = float(b0 - b1_ref*2*pi*f0 + Kg)

    A = xp.fft.fft(x_dev)

    M = NFFT*Nup - NFFT
    center = NFFT // 2 + 1
    Xc = xp.zeros(M, dtype=A.dtype)

    # Pre-compute k(z) base array on device.
    # k(z) = poling(z) * X0 * omega_abs / (4*N_mode) -- the z-dependent part
    # is just the scalar poling(z).  We pre-compute the frequency-dependent
    # part once and multiply by the scalar at each step.
    # To support arbitrary k(z), we evaluate it once to get the shape, then
    # check if it varies only by a scalar factor vs z.
    k_ref = k(0.0)
    k_ref_dev = xp.asarray(k_ref)

    # Test if k(z) = scalar(z) * k_ref  (true for QPM poling)
    k_test = k(float(L)/2)
    if np.max(np.abs(k_ref)) > 0:
        ratio = k_test / k_ref
        _k_is_separable = np.allclose(ratio, ratio[0], rtol=1e-10)
    else:
        _k_is_separable = True

    if _k_is_separable and np.max(np.abs(k_ref)) > 0:
        # k(z) = g(z) * k_shape, where k_shape is freq-dependent, g(z) is scalar
        k_shape = k_ref.copy()
        k_shape_dev = xp.asarray(k_shape)
        def k_scalar(z):
            """Return the scalar multiplier for k at position z."""
            kz = k(z)
            # Use a single element to get the scalar ratio
            idx = np.argmax(np.abs(k_ref))
            return kz[idx] / k_ref[idx]
        def k_dev_fn(z):
            return float(k_scalar(z)) * k_shape_dev
    else:
        def k_dev_fn(z):
            return xp.asarray(k(z))

    # ----- pre-allocate work buffers -----
    Aup = xp.zeros(Nup*NFFT, dtype=A.dtype)
    F1 = xp.empty(NFFT, dtype=A.dtype)

    # ----- nonlinear RHS (all on-device) -----
    def fnl(z, y):
        phi = phi_1 - phi_2*z

        y_fast = y * xp.exp(-1j*D_dev*z)

        Aup[:center] = y_fast[:center]
        Aup[center:center+M] = Xc
        Aup[center+M:] = y_fast[center:]
        aup = xp.fft.ifft(Aup) * Nup

        xup = aup * (xp.cos(phi) + 1j*xp.sin(phi))
        f1up = aup * (xup + 2*xp.conj(xup))

        F1up = xp.fft.fft(f1up)
        F1[:center] = F1up[:center]
        F1[center:] = F1up[center+M:]

        return -1j * k_dev_fn(z) * (F1 / Nup) * xp.exp(1j*D_dev*z)

    # ----- adaptive RK45 integration -----
    rtol = 1e-4
    atol = 1e-4

    z = float(z0)
    y = A.copy()
    steps = []

    # Initial step size estimate
    f0_eval = fnl(z, y)
    scale0 = atol + rtol * xp.abs(y)
    d0 = float(xp.sqrt(xp.mean(xp.abs(y / scale0)**2)))
    d1 = float(xp.sqrt(xp.mean(xp.abs(f0_eval / scale0)**2)))
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    h0 = min(h0, L - z)
    y1 = y + h0 * f0_eval
    f1_eval = fnl(z + h0, y1)
    d2 = float(xp.sqrt(xp.mean(xp.abs((f1_eval - f0_eval) / scale0)**2))) / h0
    if max(d1, d2) <= 1e-15:
        h = max(1e-6, h0 * 1e-3)
    else:
        h = min((0.01 / max(d1, d2)) ** 0.2, 100 * h0, L - z)

    # Main loop - minimize Python overhead by keeping everything on-device
    while z < L:
        if z + h > L:
            h = L - z

        # --- Dormand-Prince stages (all on-device) ---
        k0 = fnl(z, y)
        k1 = fnl(z + _C[1]*h, y + h*(1/5*k0))
        k2 = fnl(z + _C[2]*h, y + h*(3/40*k0 + 9/40*k1))
        k3 = fnl(z + _C[3]*h, y + h*(44/45*k0 - 56/15*k1 + 32/9*k2))
        k4 = fnl(z + _C[4]*h, y + h*(19372/6561*k0 - 25360/2187*k1 + 64448/6561*k2 - 212/729*k3))
        k5 = fnl(z + _C[5]*h, y + h*(9017/3168*k0 - 355/33*k1 + 46732/5247*k2 + 49/176*k3 - 5103/18656*k4))

        # 5th order solution
        y_new = y + h * (35/384*k0 + 500/1113*k2 + 125/192*k3 - 2187/6784*k4 + 11/84*k5)

        # Error estimate
        k6 = fnl(z + h, y_new)
        err = h * (71/57600*k0 - 71/16695*k2 + 71/1920*k3 - 17253/339200*k4 + 22/525*k5 - 1/40*k6)

        # Error norm (one scalar pull from device per step)
        scale = atol + rtol * xp.maximum(xp.abs(y), xp.abs(y_new))
        err_norm = float(xp.sqrt(xp.mean(xp.abs(err / scale)**2)))

        if err_norm <= 1.0:
            z += h
            y = y_new
            steps.append(h)

        # Step size control
        if err_norm > 1e-15:
            factor = min(max(0.9 * err_norm ** (-0.2), 0.2), 5.0)
        else:
            factor = 5.0
        h *= factor

    if verbose:
        print('finished')

    # ----- final transform -----
    A_out = y * xp.exp(-1j*D_dev*float(L))
    a = xp.fft.ifft(A_out)

    return to_cpu(a), steps


if __name__ == '__main__':
    pass

# -*- coding: utf-8 -*-
"""
@author: Luis Ledezma
"""
import warnings
import numpy as np
from numpy.fft import fft, ifft, fftfreq
from scipy.constants import pi, h

from scipy.integrate import RK45

def NEE(t, x, Omega, f0,
        L, D, b0, b1_ref, k,
        z0=0, verbose=True, Kg=0, Qnoise=False,
        z_save=None):
    """
    Nonlinear-envelope equation
    Adaptive solver

    Parameters
    ----------
    z_save : array-like, optional
        Sorted z-positions at which to record the time-domain field.
        When provided, the returned ``a_evol`` is a 2-D array of shape
        ``(len(z_save), NFFT)`` containing the field snapshots instead
        of the step-size array.
    """
    #Get stuff
    NFFT = t.size
    Omega_abs = Omega + 2*pi*f0
    f_max = np.amax(Omega_abs) / (2*pi)
    f_min = np.amin(Omega_abs) / (2*pi)
    BW = f_max - f_min
    Δt = 1/BW
    f = fftfreq(NFFT, Δt)
    Δf = f[1] - f[0]
    f_abs = f + f0
    
    #Calculate upsampling parameter, it usually will be Nup=4,
    #so, throw a warning if it needs to be 8
    Nup = 4
    if (3*f_max - f_min)/BW > Nup:
        Nup = 8
        warnings.warn('Large bandwidth requires 8x upsampling.', stacklevel=2)

    #Quantum noise
    if Qnoise:
        ϕ = np.random.uniform( 0, 2*pi, NFFT )
        Xnoise = BW * np.sqrt(h*f_abs/2/Δf) * np.exp(1j * ϕ)
        xnoise = ifft(Xnoise)
    else:
        xnoise = np.zeros(NFFT)

    #Input signal to frequency domain
    A = fft(x + xnoise)
    Aup = np.zeros( Nup*NFFT ) * 1j
    
    tup = np.linspace(t[0], t[-1], Nup*NFFT) #upsampled time
    phi_1 = 2*pi*f0*tup
    phi_2 = b0 - b1_ref*2*pi*f0 + Kg
    
    #Upsampling stuff
    M = NFFT*Nup - A.size
    Xc = np.zeros(M)
    center = A.size // 2 + 1

    #Nonlinear function
    def fnl(z, y):
        phi = phi_1 - phi_2*z
        
        #get fast envelope
        y = y * np.exp(-1j*D*z)
        
        #Upsample
        Aup[:center] = y[:center]
        Aup[center:center+M] = Xc
        Aup[center+M:] = y[center:]
        aup = ifft(Aup) * Nup

        #Nonlinear stuff
        xup = aup*(np.cos(phi) + 1j*np.sin(phi))
        f1up = aup*(xup + 2*np.conj(xup))

        #Downsample
        F1 = np.zeros_like(y)
        F1up = fft(f1up)
        F1[:center] = F1up[:center]
        F1[center:] = F1up[center+M:]
        F1 = F1 / Nup

        return -1j * k(z) * F1 * np.exp(1j*D*z)
    
    rtol = 1e-4
    atol = 1e-4
    
    Integrator = RK45( fnl, z0, A, L, rtol=rtol, atol=atol )

    # z-resolved field snapshots
    if z_save is not None:
        z_save = np.asarray(z_save, dtype=float)
        snapshots = np.zeros((len(z_save), NFFT), dtype=complex)
        snap_idx = 0

    steps = np.array([])
    while Integrator.status == "running":
        Integrator.step()
        steps = np.append( steps, Integrator.step_size )

        # Record snapshots at requested z positions
        if z_save is not None:
            while snap_idx < len(z_save) and Integrator.t >= z_save[snap_idx]:
                y_snap = Integrator.dense_output()(z_save[snap_idx])
                snapshots[snap_idx] = ifft(y_snap * np.exp(-1j*D*z_save[snap_idx]))
                snap_idx += 1

    if verbose:
        print( Integrator.status )

    A[:] = Integrator.y * np.exp(-1j*D*Integrator.t)
    a = ifft(A)

    if z_save is not None:
        return a, snapshots

    return a, steps


if __name__ == '__main__':
    pass
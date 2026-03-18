#!/usr/bin/env python
"""
Cross-validate the JAX GPU NEE solver against the original SciPy CPU solver.

Runs a ladder of increasingly complex physics scenarios and reports
field correlation between the two implementations.

Usage:
    python test/cross_validate.py              # standard ladder
    python test/cross_validate.py --soak       # run-until-stopped random fuzzing
    python test/cross_validate.py --soak -n 5  # 5 fuzz iterations then stop

Requires: jax.  Uses snow.nlo_scipy as the reference solver by default.
"""
import sys
import argparse
import time
import numpy as np
from scipy.constants import pi, c

from snow import pulses, waveguides, util

nm = 1e-9
um = 1e-6
mm = 1e-3
fs = 1e-15
ps = 1e-12
pJ = 1e-12
MHz = 1e6


def import_solvers(ref_path=None):
    """Import the reference (CPU) and JAX solvers."""
    if ref_path is not None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("nlo_ref", ref_path)
        nlo_ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nlo_ref)
    else:
        from snow import nlo_scipy as nlo_ref

    import snow.nlo_jax as nlo_jax

    return nlo_ref, nlo_jax


def make_grid(N=2**10, lam_start=800*nm, lam_stop=3*um):
    f_max, f_min = c / lam_start, c / lam_stop
    BW = f_max - f_min
    dt = 1 / BW
    t = -N / (2*BW) + np.arange(0, N/BW, step=dt)
    f_ref = (f_max + f_min) / 2
    return t, f_ref


def make_nee_args(t, f_ref, L, X0, alpha=0, pp=5.18*um,
                  poling_fn=None, Pavg=1e-6, tau=100*fs, lam_p=2*um):
    """Build the dict of arguments for NEE()."""
    pump = pulses.sech_pulse(t, tau, f_ref=f_ref, Pavg=Pavg,
                             f0=c/lam_p, Npwr_dB=200, frep=250*MHz)
    wg = waveguides.waveguide(w_top=1800*nm, h_thinfilm=700*nm, h_etch=350*nm)
    if poling_fn is None:
        poling_fn = lambda z: np.sign(np.cos(z * 2*pi / pp))
    wg.add_poling(poling_fn)
    wg.set_nonlinear_coeffs(N=1, X0=X0)
    wg.set_length(L)
    wg.set_loss(alpha)
    v_ref = 1 / wg.beta1(lam_p / 2)

    Omega = pump.Omega
    beta = wg.beta(pump.wl)
    D = beta - beta[0] - Omega / v_ref - 1j * alpha / 2
    omega_abs = 2*pi * pump.f0 + Omega

    def k(z):
        return wg.poling(z) * X0 * omega_abs / (4 * 1)

    args = dict(t=pump.t, x=pump.a, Omega=Omega, f0=pump.f0,
                L=L, D=D, b0=beta[0], b1_ref=1/v_ref, k=k, verbose=False)
    return pump, args


def field_correlation(a, b):
    """Normalized field overlap |<a|b>| / sqrt(<a|a><b|b>)."""
    return float(np.abs(np.sum(a * np.conj(b))) /
                 np.sqrt(np.sum(np.abs(a)**2) * np.sum(np.abs(b)**2)))


def compare(label, pump, args, nlo_ref, nlo_jax, threshold=0.999):
    """Run both solvers, compare, print result."""
    t0 = time.perf_counter()
    a_ref, s_ref = nlo_ref.NEE(**args)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    a_jax, s_jax = nlo_jax.NEE(**args)
    t_jax = time.perf_counter() - t0

    dt = pump.t[1] - pump.t[0]
    E_in = pump.energy_td()
    E_ref = float(np.sum(np.abs(a_ref)**2) * dt)
    E_jax = float(np.sum(np.abs(a_jax)**2) * dt)
    corr = field_correlation(a_ref, a_jax)

    status = 'PASS' if corr > threshold else 'WARN' if corr > 0.99 else 'FAIL'
    print(f'  {label}')
    print(f'    [{status}] corr={corr:.6f}  '
          f'steps={len(s_ref)}/{len(s_jax)} (ref/jax)  '
          f'time={t_ref:.2f}/{t_jax:.2f}s  '
          f'E_ratio={E_jax/E_in:.6f}')
    return corr, status


def run_ladder(nlo_ref, nlo_jax):
    """Standard validation ladder."""
    t, f_ref = make_grid()
    pp = 5.18 * um
    results = []

    cases = [
        ('Linear, lossless, 1mm',
         dict(L=1*mm, X0=0)),
        ('Linear, lossless, 4mm',
         dict(L=4*mm, X0=0)),
        ('Linear, 0.5dB/cm loss, 4mm',
         dict(L=4*mm, X0=0, alpha=util.absorption_coeff(0.5))),
        ('Weak NL, 1mm',
         dict(L=1*mm, X0=1.1e-12)),
        ('Full SHG, 4mm',
         dict(L=4*mm, X0=1.1e-12)),
        ('Chirped QPM, 4mm',
         dict(L=4*mm, X0=1.1e-12,
              poling_fn=lambda z: np.sign(np.cos(z*2*pi/(pp + 0.5e-6*z))))),
        ('Apodized QPM, 4mm',
         dict(L=4*mm, X0=1.1e-12,
              poling_fn=lambda z: (np.exp(-((z-2*mm)/(1*mm))**2)
                                   * np.sign(np.cos(z*2*pi/pp))))),
        ('Uniform QPM, 10mm',
         dict(L=10*mm, X0=1.1e-12)),
        ('SHG + 0.3dB/cm loss, 4mm',
         dict(L=4*mm, X0=1.1e-12, alpha=util.absorption_coeff(0.3))),
    ]

    print('=== Validation Ladder ===')
    for label, kwargs in cases:
        pump, args = make_nee_args(t, f_ref, **kwargs)
        corr, status = compare(label, pump, args, nlo_ref, nlo_jax)
        results.append((label, corr, status))

    n_pass = sum(1 for _, _, s in results if s == 'PASS')
    n_total = len(results)
    print(f'\n  {n_pass}/{n_total} passed')
    return all(s == 'PASS' for _, _, s in results)


def run_soak(nlo_ref, nlo_jax, n_iterations=None):
    """Run-until-stopped random parameter fuzzing.

    Randomly samples physically reasonable parameter combinations and
    cross-validates JAX against the reference solver.  Runs forever
    (Ctrl-C to stop) unless n_iterations is set.

    Fuzz parameters and their ranges:
      - N: 2^9, 2^10, 2^11
      - L: 0.5 - 15 mm
      - X0: 0 - 5e-12 m/V
      - alpha: 0 - 1 dB/cm
      - pp: 3 - 8 um
      - tau: 50 - 300 fs
      - Pavg: 0.1 - 50 uW
      - poling: uniform, chirped, apodized (random)
    """
    print('=== Soak Test (Ctrl-C to stop) ===')
    rng = np.random.default_rng()
    iteration = 0
    n_pass = 0
    n_warn = 0
    n_fail = 0
    worst_corr = 1.0
    worst_label = ''

    try:
        while n_iterations is None or iteration < n_iterations:
            iteration += 1

            # Random parameters
            log2N = rng.choice([9, 10, 11])
            N = 2**log2N
            L = rng.uniform(0.5, 15) * mm
            X0 = rng.uniform(0, 5e-12)
            alpha_dBcm = rng.uniform(0, 1.0)
            alpha = util.absorption_coeff(alpha_dBcm) if alpha_dBcm > 0.01 else 0
            pp = rng.uniform(3, 8) * um
            tau = rng.uniform(50, 300) * fs
            Pavg = rng.uniform(0.1, 50) * 1e-6
            lam_p = 2 * um

            # Random poling type
            poling_type = rng.choice(['uniform', 'chirped', 'apodized'])
            if poling_type == 'uniform':
                poling_fn = lambda z, _pp=pp: np.sign(np.cos(z*2*pi/_pp))
            elif poling_type == 'chirped':
                chirp = rng.uniform(-1, 1) * 1e-6
                poling_fn = lambda z, _pp=pp, _c=chirp: np.sign(
                    np.cos(z*2*pi/(_pp + _c*z)))
            else:
                poling_fn = lambda z, _pp=pp, _L=L: (
                    np.exp(-((z - _L/2)/(_L/4))**2)
                    * np.sign(np.cos(z*2*pi/_pp)))

            label = (f'#{iteration} N=2^{log2N} L={L/mm:.1f}mm '
                     f'X0={X0:.1e} a={alpha_dBcm:.1f}dB/cm '
                     f'pp={pp/um:.1f}um tau={tau/fs:.0f}fs '
                     f'P={Pavg*1e6:.1f}uW {poling_type}')

            t, f_ref = make_grid(N=N)
            pump, args = make_nee_args(t, f_ref, L=L, X0=X0, alpha=alpha,
                                       pp=pp, poling_fn=poling_fn,
                                       Pavg=Pavg, tau=tau, lam_p=lam_p)

            corr, status = compare(label, pump, args, nlo_ref, nlo_jax,
                                   threshold=0.99)

            if status == 'PASS':
                n_pass += 1
            elif status == 'WARN':
                n_warn += 1
            else:
                n_fail += 1

            if corr < worst_corr:
                worst_corr = corr
                worst_label = label

            if iteration % 10 == 0 or status == 'FAIL':
                print(f'\n  --- After {iteration} iterations: '
                      f'{n_pass} pass, {n_warn} warn, {n_fail} fail, '
                      f'worst={worst_corr:.6f} ---\n')

    except KeyboardInterrupt:
        print('\n  Stopped by user.')

    print(f'\n  Final: {iteration} iterations, '
          f'{n_pass} pass, {n_warn} warn, {n_fail} fail')
    print(f'  Worst correlation: {worst_corr:.6f}')
    print(f'    {worst_label}')
    return n_fail == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', default=None,
                        help='Path to reference NEE solver (default: snow.nlo_scipy)')
    parser.add_argument('--soak', action='store_true',
                        help='Run random parameter fuzzing instead of fixed ladder')
    parser.add_argument('-n', type=int, default=None,
                        help='Number of soak iterations (default: run until Ctrl-C)')
    args = parser.parse_args()

    nlo_ref, nlo_jax = import_solvers(args.reference)

    if args.soak:
        ok = run_soak(nlo_ref, nlo_jax, n_iterations=args.n)
    else:
        ok = run_ladder(nlo_ref, nlo_jax)

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

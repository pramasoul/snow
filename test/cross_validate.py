#!/usr/bin/env python
"""
Cross-validate the JAX GPU NEE solver against the original SciPy CPU solver.

Runs a ladder of increasingly complex physics scenarios and reports
field correlation between the two implementations.

Usage:
    python test/cross_validate.py                        # standard ladder
    python test/cross_validate.py --soak                 # run-until-stopped fuzzing
    python test/cross_validate.py --soak -n 5            # 5 fuzz iterations
    python test/cross_validate.py --soak --log results   # log to results.jsonl

Requires: jax.  Uses snow.nlo_scipy as the reference solver by default.
"""
import sys
import json
import argparse
import time
from datetime import datetime, timezone
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


class JSONLLogger:
    """Append-only JSONL logger."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, 'a')
        # Write a session header
        self.write_record({
            'event': 'session_start',
            'timestamp': self._now(),
            'python': sys.version.split()[0],
            'argv': sys.argv,
        })

    def _now(self):
        return datetime.now(timezone.utc).isoformat()

    def write_record(self, record):
        record.setdefault('timestamp', self._now())
        self.f.write(json.dumps(record, default=str) + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


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


def compare(label, pump, args, nlo_ref, nlo_jax, threshold=0.999,
            logger=None, params=None):
    """Run both solvers, compare, print result, optionally log."""
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

    if logger is not None:
        record = {
            'event': 'comparison',
            'label': label,
            'status': status,
            'correlation': corr,
            'steps_ref': len(s_ref),
            'steps_jax': len(s_jax),
            'time_ref_s': round(t_ref, 4),
            'time_jax_s': round(t_jax, 4),
            'speedup': round(t_ref / t_jax, 2) if t_jax > 0 else None,
            'E_in': E_in,
            'E_ref': E_ref,
            'E_jax': E_jax,
            'E_ratio_ref': E_ref / E_in,
            'E_ratio_jax': E_jax / E_in,
        }
        if params is not None:
            record['params'] = params
        logger.write_record(record)

    return corr, status


def run_ladder(nlo_ref, nlo_jax, logger=None):
    """Standard validation ladder."""
    t, f_ref = make_grid()
    pp = 5.18 * um
    results = []

    cases = [
        ('Linear, lossless, 1mm',
         dict(L=1*mm, X0=0),
         dict(L_mm=1, X0=0, alpha_dBcm=0, poling='none')),
        ('Linear, lossless, 4mm',
         dict(L=4*mm, X0=0),
         dict(L_mm=4, X0=0, alpha_dBcm=0, poling='none')),
        ('Linear, 0.5dB/cm loss, 4mm',
         dict(L=4*mm, X0=0, alpha=util.absorption_coeff(0.5)),
         dict(L_mm=4, X0=0, alpha_dBcm=0.5, poling='none')),
        ('Weak NL, 1mm',
         dict(L=1*mm, X0=1.1e-12),
         dict(L_mm=1, X0=1.1e-12, alpha_dBcm=0, poling='uniform', pp_um=5.18)),
        ('Full SHG, 4mm',
         dict(L=4*mm, X0=1.1e-12),
         dict(L_mm=4, X0=1.1e-12, alpha_dBcm=0, poling='uniform', pp_um=5.18)),
        ('Chirped QPM, 4mm',
         dict(L=4*mm, X0=1.1e-12,
              poling_fn=lambda z: np.sign(np.cos(z*2*pi/(pp + 0.5e-6*z)))),
         dict(L_mm=4, X0=1.1e-12, alpha_dBcm=0, poling='chirped',
              pp_um=5.18, chirp_um_per_mm=0.5)),
        ('Apodized QPM, 4mm',
         dict(L=4*mm, X0=1.1e-12,
              poling_fn=lambda z: (np.exp(-((z-2*mm)/(1*mm))**2)
                                   * np.sign(np.cos(z*2*pi/pp)))),
         dict(L_mm=4, X0=1.1e-12, alpha_dBcm=0, poling='apodized', pp_um=5.18)),
        ('Uniform QPM, 10mm',
         dict(L=10*mm, X0=1.1e-12),
         dict(L_mm=10, X0=1.1e-12, alpha_dBcm=0, poling='uniform', pp_um=5.18)),
        ('SHG + 0.3dB/cm loss, 4mm',
         dict(L=4*mm, X0=1.1e-12, alpha=util.absorption_coeff(0.3)),
         dict(L_mm=4, X0=1.1e-12, alpha_dBcm=0.3, poling='uniform', pp_um=5.18)),
    ]

    print('=== Validation Ladder ===')
    for label, kwargs, params in cases:
        params['mode'] = 'ladder'
        params['N'] = 1024
        pump, args = make_nee_args(t, f_ref, **kwargs)
        corr, status = compare(label, pump, args, nlo_ref, nlo_jax,
                               logger=logger, params=params)
        results.append((label, corr, status))

    n_pass = sum(1 for _, _, s in results if s == 'PASS')
    n_total = len(results)
    print(f'\n  {n_pass}/{n_total} passed')

    if logger:
        logger.write_record({
            'event': 'ladder_summary',
            'n_pass': n_pass,
            'n_total': n_total,
        })

    return all(s == 'PASS' for _, _, s in results)


def run_soak(nlo_ref, nlo_jax, n_iterations=None, logger=None):
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
            log2N = int(rng.choice([9, 10, 11]))
            N = 2**log2N
            L = float(rng.uniform(0.5, 15)) * mm
            X0 = float(rng.uniform(0, 5e-12))
            alpha_dBcm = float(rng.uniform(0, 1.0))
            alpha = util.absorption_coeff(alpha_dBcm) if alpha_dBcm > 0.01 else 0
            pp = float(rng.uniform(3, 8)) * um
            tau = float(rng.uniform(50, 300)) * fs
            Pavg = float(rng.uniform(0.1, 50)) * 1e-6
            lam_p = 2 * um

            # Random poling type
            poling_type = str(rng.choice(['uniform', 'chirped', 'apodized']))
            chirp = 0.0
            if poling_type == 'uniform':
                poling_fn = lambda z, _pp=pp: np.sign(np.cos(z*2*pi/_pp))
            elif poling_type == 'chirped':
                chirp = float(rng.uniform(-1, 1)) * 1e-6
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

            params = {
                'mode': 'soak',
                'iteration': iteration,
                'N': N,
                'log2N': log2N,
                'L_mm': L / mm,
                'X0': X0,
                'alpha_dBcm': alpha_dBcm,
                'pp_um': pp / um,
                'tau_fs': tau / fs,
                'Pavg_uW': Pavg * 1e6,
                'poling': poling_type,
                'chirp_um_per_mm': chirp / 1e-6 if poling_type == 'chirped' else None,
            }

            t, f_ref = make_grid(N=N)
            pump, args = make_nee_args(t, f_ref, L=L, X0=X0, alpha=alpha,
                                       pp=pp, poling_fn=poling_fn,
                                       Pavg=Pavg, tau=tau, lam_p=lam_p)

            corr, status = compare(label, pump, args, nlo_ref, nlo_jax,
                                   threshold=0.99,
                                   logger=logger, params=params)

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
                summary = (f'{n_pass} pass, {n_warn} warn, {n_fail} fail, '
                           f'worst={worst_corr:.6f}')
                print(f'\n  --- After {iteration} iterations: {summary} ---\n')
                if logger:
                    logger.write_record({
                        'event': 'soak_progress',
                        'iteration': iteration,
                        'n_pass': n_pass,
                        'n_warn': n_warn,
                        'n_fail': n_fail,
                        'worst_corr': worst_corr,
                        'worst_label': worst_label,
                    })

    except KeyboardInterrupt:
        print('\n  Stopped by user.')

    print(f'\n  Final: {iteration} iterations, '
          f'{n_pass} pass, {n_warn} warn, {n_fail} fail')
    print(f'  Worst correlation: {worst_corr:.6f}')
    print(f'    {worst_label}')

    if logger:
        logger.write_record({
            'event': 'soak_summary',
            'iterations': iteration,
            'n_pass': n_pass,
            'n_warn': n_warn,
            'n_fail': n_fail,
            'worst_corr': worst_corr,
            'worst_label': worst_label,
        })

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
    parser.add_argument('--log', metavar='FILE', default=None,
                        help='Log results to FILE.jsonl (appends if exists)')
    args = parser.parse_args()

    logger = None
    if args.log:
        logpath = args.log if args.log.endswith('.jsonl') else args.log + '.jsonl'
        logger = JSONLLogger(logpath)
        print(f'Logging to {logpath}')

    nlo_ref, nlo_jax = import_solvers(args.reference)

    try:
        if args.soak:
            ok = run_soak(nlo_ref, nlo_jax, n_iterations=args.n, logger=logger)
        else:
            ok = run_ladder(nlo_ref, nlo_jax, logger=logger)
    finally:
        if logger:
            logger.close()

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

"""Regression harness: run the same MUSE case with the frozen v1 solver and the
current (v2) solver in two subprocesses and compare the sensor signals.

Metrics per sensor: absolute RMSE, RMS-normalised NRMSE, peak ratio, and the
NRMSE restricted to a direct-arrival window (before edge reflections arrive).

Usage::

    python tests/compare_v1_v2.py --v1-dir <v1 snapshot> [--v2-dir <repo root>]
        [--case dmg|intact] [--nrays 201] [--nt 5000] [--tmax 1e-4] [--nfft 250]
        [--out-dir <dir>] [--skip-run]

The v1 snapshot is the repository root at commit 9b00afb (``git archive``)
with ``Beam.__init__`` patched to the v2 burst normalisation
(``signal_f(self.t, 1.0, f, fd)``) so that both versions scale identically.
"""
import argparse
import json
import os
import subprocess
import sys

import h5py
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)


def run(code_dir: str, out: str, args: argparse.Namespace, log: str,
        extra: tuple = ()) -> dict:
    """Run ``run_case.py`` for one solver directory and return its timing dict."""
    cmd = [sys.executable, os.path.join(_THIS_DIR, 'run_case.py'),
           '--code-dir', code_dir, '--out', out, '--case', args.case,
           '--nrays', str(args.nrays), '--nt', str(args.nt), '--tmax', str(args.tmax),
           '--nfft', str(args.nfft), '--source', str(args.source), '--log', log,
           *extra]
    res = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout + res.stderr)
        raise SystemExit('run failed for {}'.format(code_dir))
    return json.loads(res.stdout.strip().splitlines()[-1])


def load(fn: str) -> tuple:
    """Return (time, signal matrix, sensor names) from a run_case output file."""
    with h5py.File(fn, 'r') as f:
        t = f['time'][:]
        keys = [k for k in f if k != 'time']
        grp = f[keys[0]]
        dname = list(grp.keys())[0]
        d = grp[dname]
        return t, d[:], list(d.attrs['columns'])


def metrics(t: np.ndarray, ref: np.ndarray, new: np.ndarray, cols: list,
            win: tuple) -> None:
    """Print per-sensor comparison metrics of ``new`` against ``ref``."""
    w = (t >= win[0]) & (t <= win[1])
    print('{:>6s} {:>12s} {:>12s} {:>10s} {:>14s}'.format(
        'sensor', 'rmse_abs', 'nrmse_rms', 'peak_ratio', 'nrmse_window'))
    for i, c in enumerate(cols):
        d = new[:, i] - ref[:, i]
        rms = np.sqrt(np.mean(ref[:, i] ** 2))
        rmse = np.sqrt(np.mean(d ** 2))
        rms_w = np.sqrt(np.mean(ref[w, i] ** 2))
        nrmse_w = np.sqrt(np.mean(d[w] ** 2)) / rms_w if rms_w > 0 else np.nan
        pk_ref = np.abs(ref[:, i]).max()
        pk = np.abs(new[:, i]).max() / pk_ref if pk_ref > 0 else np.nan
        print('{:>6s} {:12.3e} {:12.3e} {:10.6f} {:14.3e}'.format(
            c, rmse, rmse / rms if rms > 0 else np.nan, pk, nrmse_w))
    d = new - ref
    print('{:>6s} {:12.3e} {:12.3e}'.format(
        'ALL', np.sqrt(np.mean(d ** 2)),
        np.sqrt(np.mean(d ** 2)) / np.sqrt(np.mean(ref ** 2))))


def main() -> None:
    """Run both solvers (unless --skip-run) and print the comparison."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--v1-dir', required=True)
    ap.add_argument('--v2-dir', default=_REPO_ROOT)
    ap.add_argument('--case', default='dmg', choices=['dmg', 'intact'])
    ap.add_argument('--nrays', type=int, default=201)
    ap.add_argument('--nt', type=int, default=5000)
    ap.add_argument('--tmax', type=float, default=1.e-4)
    ap.add_argument('--nfft', type=int, default=250)
    ap.add_argument('--source', type=int, default=0)
    ap.add_argument('--out-dir', default=os.path.join(_THIS_DIR, 'out'))
    ap.add_argument('--window', type=float, nargs=2, default=(20.e-6, 50.e-6),
                    help='direct-arrival window [s] for the windowed NRMSE')
    ap.add_argument('--skip-run', action='store_true',
                    help='only compare existing outputs in --out-dir')
    ap.add_argument('--only', choices=['v1', 'v2'], default=None,
                    help='run only one of the two solvers (then compare)')
    ap.add_argument('--v2-birth-dispersion', action='store_true',
                    help='run v2 with the birth-direction dispersion curve kept after '
                         'reflections (v1 physics); output file gets suffix _birth')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = '{}_n{}_nt{}_nfft{}'.format(args.case, args.nrays, args.nt, args.nfft)
    f1 = os.path.join(args.out_dir, 'v1_{}.hdf5'.format(tag))
    v2_tag = tag + ('_birth' if args.v2_birth_dispersion else '')
    f2 = os.path.join(args.out_dir, 'v2_{}.hdf5'.format(v2_tag))
    v2_extra = ('--birth-direction-dispersion',) if args.v2_birth_dispersion else ()

    if not args.skip_run:
        r1 = r2 = None
        if args.only in (None, 'v1'):
            r1 = run(args.v1_dir, f1, args, f1.replace('.hdf5', '.log'))
            print('v1:', json.dumps(r1))
        if args.only in (None, 'v2'):
            r2 = run(args.v2_dir, f2, args, f2.replace('.hdf5', '.log'), v2_extra)
            print('v2:', json.dumps(r2))
        if r1 and r2 and r2['t_total'] > 0:
            print('speed-up (total): {:.1f}x'.format(r1['t_total'] / r2['t_total']))

    t, ref, cols = load(f1)
    _, new, cols2 = load(f2)
    assert cols == cols2, (cols, cols2)
    print('signal RMS (v1): {:.3e}   peak (v1): {:.3e}'.format(
        np.sqrt(np.mean(ref ** 2)), np.abs(ref).max()))
    metrics(t, ref, new, cols, tuple(args.window))


if __name__ == '__main__':
    main()

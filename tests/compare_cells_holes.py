"""Compare the 2-medium hole mesh of ``gen_MUSE_dmg`` (``mesh='holes'``) against
the legacy convex-cell mesh with invisible walls (``mesh='cells'``), both run
with the current solver in two subprocesses of ``run_case.py``.

The two meshes describe the same physics; the only modelling difference is
that the cell mesh kills and re-spawns every ray 1e-8 mm past each invisible
wall, so the signals are expected to agree to roughly that level.

Metrics per sensor: absolute RMSE, RMS-normalised NRMSE, max abs difference,
peak ratio and the first-arrival sample of each run (first |s| above 1 % of
the peak).

Usage::

    python tests/compare_cells_holes.py [--nrays 201] [--nt 5000] [--tmax 1e-4]
        [--nfft 250] [--source 0] [--shape rect|ellipse] [--phidmg rad]
        [--full] [--birth-direction-dispersion] [--plot] [--out-dir <dir>]
        [--skip-run]
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


def run(mesh: str, out: str, args: argparse.Namespace, log: str) -> dict:
    """Run ``run_case.py`` for one mesh and return its timing dict."""
    cmd = [sys.executable, os.path.join(_THIS_DIR, 'run_case.py'),
           '--code-dir', _REPO_ROOT, '--out', out, '--case', 'dmg',
           '--nrays', str(args.nrays), '--nt', str(args.nt), '--tmax', str(args.tmax),
           '--nfft', str(args.nfft), '--source', str(args.source), '--log', log,
           '--mesh', mesh, '--shape', args.shape, '--phidmg', str(args.phidmg)]
    if args.birth_direction_dispersion:
        cmd.append('--birth-direction-dispersion')
    res = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout + res.stderr)
        raise SystemExit('run failed for mesh={}'.format(mesh))
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


def first_arrival(s: np.ndarray, frac: float = 0.01) -> int:
    """Index of the first sample whose magnitude exceeds ``frac`` of the peak (-1 if silent)."""
    pk = np.abs(s).max()
    if pk <= 0.:
        return -1
    return int(np.argmax(np.abs(s) > frac * pk))


def metrics(t: np.ndarray, ref: np.ndarray, new: np.ndarray, cols: list) -> float:
    """Print per-sensor metrics of ``new`` (holes) against ``ref`` (cells); return the worst NRMSE."""
    print('{:>6s} {:>12s} {:>12s} {:>12s} {:>10s} {:>9s} {:>9s}'.format(
        'sensor', 'rmse_abs', 'nrmse_rms', 'max_abs_diff', 'peak_ratio', 'i0_cells', 'i0_holes'))
    worst = 0.
    for i, c in enumerate(cols):
        d = new[:, i] - ref[:, i]
        rms = np.sqrt(np.mean(ref[:, i] ** 2))
        rmse = np.sqrt(np.mean(d ** 2))
        nrmse = rmse / rms if rms > 0 else np.nan
        pk_ref = np.abs(ref[:, i]).max()
        pk = np.abs(new[:, i]).max() / pk_ref if pk_ref > 0 else np.nan
        if np.isfinite(nrmse):
            worst = max(worst, nrmse)
        print('{:>6s} {:12.3e} {:12.3e} {:12.3e} {:10.6f} {:9d} {:9d}'.format(
            c, rmse, nrmse, np.abs(d).max(), pk,
            first_arrival(ref[:, i]), first_arrival(new[:, i])))
    d = new - ref
    print('{:>6s} {:12.3e} {:12.3e} {:12.3e}'.format(
        'ALL', np.sqrt(np.mean(d ** 2)),
        np.sqrt(np.mean(d ** 2)) / np.sqrt(np.mean(ref ** 2)), np.abs(d).max()))
    return worst


def plot(t: np.ndarray, ref: np.ndarray, new: np.ndarray, cols: list) -> None:
    import matplotlib.pyplot as plt
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
    for i, (ax, c) in enumerate(zip(np.atleast_1d(axes), cols)):
        ax.plot(t * 1e6, ref[:, i], 'k-', lw=1., label='cells')
        ax.plot(t * 1e6, new[:, i], 'r--', lw=0.8, label='holes')
        ax2 = ax.twinx()
        ax2.plot(t * 1e6, new[:, i] - ref[:, i], 'b-', lw=0.5, alpha=0.6)
        ax2.set_ylabel('diff', color='b')
        ax.set_ylabel(c)
        if i == 0:
            ax.legend(loc='upper right')
    np.atleast_1d(axes)[-1].set_xlabel('t [us]')
    fig.tight_layout()
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--nrays', type=int, default=201)
    ap.add_argument('--nt', type=int, default=5000)
    ap.add_argument('--tmax', type=float, default=1.e-4)
    ap.add_argument('--nfft', type=int, default=250)
    ap.add_argument('--source', type=int, default=0)
    ap.add_argument('--shape', default='rect', choices=['rect', 'ellipse'])
    ap.add_argument('--phidmg', type=float, default=0.)
    ap.add_argument('--full', action='store_true',
                    help='full benchmark: 2001 rays, 10000 samples, nfft 1000')
    ap.add_argument('--birth-direction-dispersion', action='store_true')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--out-dir', default=os.path.join(_THIS_DIR, 'out'))
    ap.add_argument('--skip-run', action='store_true',
                    help='only compare existing outputs in --out-dir')
    args = ap.parse_args()
    if args.full:
        args.nrays, args.nt, args.nfft = 2001, 10000, 1000

    os.makedirs(args.out_dir, exist_ok=True)
    tag = '{}_n{}_nt{}_nfft{}{}'.format(args.shape, args.nrays, args.nt, args.nfft,
                                       '_birth' if args.birth_direction_dispersion else '')
    files = {m: os.path.join(args.out_dir, '{}_{}.hdf5'.format(m, tag)) for m in ('cells', 'holes')}

    if not args.skip_run:
        res = {}
        for mesh, fn in files.items():
            res[mesh] = run(mesh, fn, args, fn.replace('.hdf5', '.log'))
            print('{:>5s}: {}'.format(mesh, json.dumps(res[mesh])))
        if res['holes']['t_total'] > 0:
            print('rays traced: cells {} / holes {}   speed-up (total): {:.2f}x'.format(
                res['cells']['n_rays_traced'], res['holes']['n_rays_traced'],
                res['cells']['t_total'] / res['holes']['t_total']))

    t, ref, cols = load(files['cells'])
    _, new, cols2 = load(files['holes'])
    assert cols == cols2, (cols, cols2)
    print('reference (cells): {}'.format(files['cells']))
    print('signal RMS (ref): {:.3e}   peak (ref): {:.3e}'.format(
        np.sqrt(np.mean(ref ** 2)), np.abs(ref).max()))
    worst = metrics(t, ref, new, cols)
    print('worst per-sensor NRMSE: {:.3e}'.format(worst))
    if args.plot:
        plot(t, ref, new, cols)


if __name__ == '__main__':
    main()

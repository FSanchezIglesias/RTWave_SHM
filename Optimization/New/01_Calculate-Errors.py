"""Compare the sensor signals of each optimisation stage with the reference run.

For every ``<name>.hdf5`` in ``optimization/`` (except ``ref.hdf5``) prints:

* ``rmse_abs``  - root-mean-square error over all sensors and time samples
                  (this is the quantity that was historically logged as
                  "error" in ``Performance_Evolution.csv``);
* ``nrmse_rms`` - the same error divided by the RMS value of the reference
                  signals (the NRMSE quoted in the thesis table);
* ``max_abs``   - largest absolute deviation of any sample.

Usage::

    python 01_Calculate-Errors.py [--dir optimization] [--key DMG12_355.00_99.00/PZT1] [--plot]
"""
import argparse
import glob
import os

import h5py
import numpy as np


def load(fname: str, key: str) -> np.ndarray:
    """Return the ``(n_time, n_sensors)`` signal matrix stored under ``key``."""
    with h5py.File(fname, 'r') as h5f:
        return h5f[key][:]


def main() -> None:
    """Print the error metrics of every stage file against the reference."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='optimization', help='folder with the stage files')
    ap.add_argument('--ref', default='ref.hdf5', help='reference file name inside --dir')
    ap.add_argument('--key', default='DMG12_355.00_99.00/PZT1')
    ap.add_argument('--plot', action='store_true', help='plot the difference of each stage')
    args = ap.parse_args()

    ref = load(os.path.join(args.dir, args.ref), args.key)
    rms_ref = np.sqrt(np.mean(ref ** 2))
    print('reference: {}  rms={:.4e}  peak={:.4e}'.format(args.ref, rms_ref, np.abs(ref).max()))
    print('{:16s} {:>12s} {:>12s} {:>12s}'.format('stage', 'rmse_abs', 'nrmse_rms', 'max_abs'))

    files = sorted(glob.glob(os.path.join(args.dir, '*.hdf5')))
    for fname in files:
        if os.path.basename(fname) == args.ref:
            continue
        sig = load(fname, args.key)
        diff = sig - ref
        rmse = np.sqrt(np.mean(diff ** 2))
        print('{:16s} {:12.3e} {:12.3e} {:12.3e}'.format(
            os.path.basename(fname), rmse, rmse / rms_ref, np.abs(diff).max()))

        if args.plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(diff)
            plt.title(os.path.basename(fname))
    if args.plot:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main()

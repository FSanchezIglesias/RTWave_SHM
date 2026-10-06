"""Run one MUSE ray-tracing case with a given solver code directory.

Used by ``compare_v1_v2.py``: the same script drives either the frozen v1
snapshot or the current (v2) package so that the two outputs can be compared.
Writes the sensor signals to an HDF5 file and prints a JSON line with timings.

Usage::

    python tests/run_case.py --code-dir <dir with geom/ RayTracing/ utils_rays/>
                             --out <file.hdf5> [--case dmg|intact] [--nrays N]
                             [--nt N] [--tmax s] [--nfft N] [--source i]
                             [--mesh holes|cells] [--shape rect|ellipse]
"""
import argparse
import json
import logging
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_EXAMPLE_DIR = os.path.join(_REPO_ROOT, 'example')


def main() -> None:
    """Parse arguments, run the case and write the signals to HDF5."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--code-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--case', default='dmg', choices=['dmg', 'intact'])
    ap.add_argument('--nrays', type=int, default=201)
    ap.add_argument('--nt', type=int, default=5000)
    ap.add_argument('--tmax', type=float, default=1.e-4)
    ap.add_argument('--nfft', type=int, default=250)
    ap.add_argument('--source', type=int, default=0)
    ap.add_argument('--xdmg', type=float, default=355.)
    ap.add_argument('--ydmg', type=float, default=99.)
    ap.add_argument('--log', default=None)
    ap.add_argument('--mesh', default='holes', choices=['holes', 'cells'],
                    help="gen_MUSE_dmg mesh: 'holes' (2 mediums) or the legacy"
                         " convex-cell mesh with invisible walls")
    ap.add_argument('--shape', default='rect', choices=['rect', 'ellipse'])
    ap.add_argument('--phidmg', type=float, default=0.,
                    help='ellipse rotation [rad] (shape=ellipse only)')
    ap.add_argument('--birth-direction-dispersion', action='store_true',
                    help='keep the birth-direction dispersion curve after reflections '
                         '(v1 behaviour); sets RayTracing.Ray.dispersion_follows_direction=False')
    ap.add_argument('--no-diffraction', action='store_true',
                    help='disable the corner diffraction (RayTracing.Ray.diffraction=False); '
                         'ignored by solver copies that do not have it')
    ap.add_argument('--v1-physics', action='store_true',
                    help='v1 wall physics: no corner diffraction and the transmitted share is '
                         'lost at total internal reflection (RayTracing.Ray.diffraction=False, '
                         'total_internal_reflection=False); ignored by copies without them')
    args = ap.parse_args()

    # Bind the solver package to the requested directory BEFORE anything from
    # example/ (which prepends the repo root to sys.path) is imported.
    sys.path.insert(0, os.path.abspath(args.code_dir))
    import geom.map_2d  # noqa: F401,E402
    import RayTracing.Ray  # noqa: E402
    import RayTracing.Sensors  # noqa: F401,E402
    import utils_rays.ray_utils  # noqa: F401,E402
    from RayTracing.Ray import Beam_from_pzt  # noqa: E402

    if args.birth_direction_dispersion:
        RayTracing.Ray.dispersion_follows_direction = False
    if (args.no_diffraction or args.v1_physics) and hasattr(RayTracing.Ray, 'diffraction'):
        RayTracing.Ray.diffraction = False
    if args.v1_physics and hasattr(RayTracing.Ray, 'total_internal_reflection'):
        RayTracing.Ray.total_internal_reflection = False

    sys.path.insert(0, _EXAMPLE_DIR)
    import numpy as np  # noqa: E402
    import h5py  # noqa: E402
    from MUSE_dmg import gen_MUSE_dmg, gen_MUSE_intact  # noqa: E402
    from plate_config import XLDMG, YLDMG, THDMG, RDMG, BLDMG, RMDMG  # noqa: E402

    if args.log:
        logging.basicConfig(filename=args.log, filemode='w', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    t = np.linspace(0., args.tmax, args.nt)
    if args.case == 'dmg':
        m, pzts = gen_MUSE_dmg(xdmg=args.xdmg, ydmg=args.ydmg, xldmg=XLDMG, yldmg=YLDMG,
                               thdmg=THDMG, rdmg=RDMG, bldmg=BLDMG, rmdmg=RMDMG,
                               shape=args.shape, phidmg=args.phidmg, mesh=args.mesh)
        key = 'DMG12_{:.2f}_{:.2f}/PZT{}'.format(args.xdmg, args.ydmg, args.source + 1)
    else:
        m, pzts = gen_MUSE_intact()
        key = 'intact/PZT{}'.format(args.source + 1)

    beam = Beam_from_pzt(args.nrays, pzts[args.source], power=args.nrays / len(pzts),
                         f=350.e3, npeaks=3, nfft=args.nfft, t=t)
    m.set_init_beam(beam)

    t0 = time.perf_counter()
    m.calc_t()
    t1 = time.perf_counter()
    m.calc_signal()
    t2 = time.perf_counter()

    if os.path.exists(args.out):
        os.remove(args.out)
    with h5py.File(args.out, 'a') as h5f:
        h5f['time'] = t
    m.save_signals(args.out, key)
    m.close_h5()

    print(json.dumps({'code_dir': os.path.abspath(args.code_dir), 'key': key,
                      'mesh': args.mesh, 'n_mediums': len(m.mediums),
                      'n_rays_traced': len(m.rays_h),
                      't_calc_t': t1 - t0, 't_calc_signal': t2 - t1,
                      't_total': t2 - t0}))


if __name__ == '__main__':
    main()

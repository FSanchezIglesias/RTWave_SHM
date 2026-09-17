"""Tests of the nearest-hit ray tracing (``Ray.trace``) and of the 2-medium
hole mesh of ``example/MUSE_dmg.py`` (``mesh='holes'``) against the legacy
convex-cell mesh with invisible walls (``mesh='cells'``).

Run from the repository root::

    python tests/test_hole_mesh.py
"""
import os
import sys
import unittest

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLE_DIR = os.path.join(_REPO_ROOT, 'example')
for _p in (_REPO_ROOT, _EXAMPLE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from geom.objects_2d import Segment, Ellipse, medium  # noqa: E402
from geom.map_2d import Map2D  # noqa: E402


def _nrmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2))


class _ConstWS:
    """Isotropic, non-dispersive wave-speed model [m/s]."""

    def __init__(self, v):
        self._v = float(v)

    def S0(self, fa):
        return self._v

    def A0(self, fa):
        return self._v


class TestNearestHit(unittest.TestCase):

    def test_trace_interacts_with_nearest_wall(self):
        # Non-convex "L"-free case: a square medium plus a hole (inner square)
        # whose walls are listed AFTER the outer walls.  A ray shot from the
        # left towards the hole must hit the hole's left wall, not the far
        # right outer wall that the full-length trace also crosses.
        from RayTracing.Ray import Ray
        from scipy.fft import rfftfreq

        kw = dict(boundary_losses=0., ratio_rfl=1., ratio_mode=1.)
        outer = [Segment(np.array([0., 0.]), np.array([100., 0.]), **kw),
                 Segment(np.array([100., 0.]), np.array([100., 100.]), **kw),
                 Segment(np.array([100., 100.]), np.array([0., 100.]), **kw),
                 Segment(np.array([0., 100.]), np.array([0., 0.]), **kw)]
        hole = [Segment(np.array([40., 40.]), np.array([60., 40.]), **kw),
                Segment(np.array([60., 40.]), np.array([60., 60.]), **kw),
                Segment(np.array([60., 60.]), np.array([40., 60.]), **kw),
                Segment(np.array([40., 60.]), np.array([40., 40.]), **kw)]
        m = medium(_ConstWS(1000.), 1., xi=0., dispersive=False)  # 1e6 mm/s
        m.add_objs(outer + hole)  # far walls first on purpose
        r_map = Map2D(mediums=[m])

        nfft = 64
        t = np.linspace(0., 1.e-4, 200)
        freq = np.ones(nfft // 2 + 1, dtype=complex)
        fft_freq = rfftfreq(nfft, d=t[1] - t[0])
        ray = Ray(np.array([10., 50.]), np.array([1., 0.]), freq=freq, medium=m,
                  t=t, t0=0., kind='S0', a=1., _fft_freq=fft_freq)
        r_map.save_ray(ray)
        r_map.rays_h = [ray.__hash__()]
        r_map.calc_t(t=1.e-4)  # 1e6 mm/s * 1e-4 s = 100 mm of travel

        # first event after birth is the intersection with the hole wall at x=40
        self.assertGreaterEqual(len(ray.trace_points), 2)
        np.testing.assert_allclose(ray.trace_points[1][0], 40., atol=1e-6)
        np.testing.assert_allclose(ray.trace_points[1][1], 50., atol=1e-6)
        # ... and it bounced back towards the left wall
        self.assertLess(ray.d[1][0], 0.)

    def test_hit_returns_distance_and_oriented_normal(self):
        seg = Segment(np.array([5., -1.]), np.array([5., 1.]))
        s, p, n, d = seg.hit(np.array([0., 0.]), np.array([10., 0.]))
        self.assertAlmostEqual(s, 5.)
        np.testing.assert_allclose(p, [5., 0.], atol=1e-12)
        np.testing.assert_allclose(n, [1., 0.], atol=1e-12)  # away from the ray origin
        s2, _, n2, _ = seg.hit(np.array([10., 0.]), np.array([0., 0.]))
        self.assertAlmostEqual(s2, 5.)
        np.testing.assert_allclose(n2, [-1., 0.], atol=1e-12)
        self.assertIsNone(seg.hit(np.array([0., 5.]), np.array([10., 5.])))

        e = Ellipse([0., 0.], 2., 1.)
        s, p, n, d = e.hit(np.array([-5., 0.]), np.array([5., 0.]))
        self.assertAlmostEqual(s, 3.)
        np.testing.assert_allclose(n, [1., 0.], atol=1e-12)  # inward: ray comes from outside


class TestMuseHoleMesh(unittest.TestCase):
    """End-to-end checks of gen_MUSE_dmg(mesh='holes') with a reduced beam."""

    N_RAYS = 201
    NFFT = 250
    T = np.linspace(0., 0.0001, 2500)

    @classmethod
    def _run(cls, **kw):
        from MUSE_dmg import gen_MUSE_dmg, gen_MUSE_intact
        from RayTracing.Ray import Beam_from_pzt
        if kw.pop('intact', False):
            m, pzts = gen_MUSE_intact()
        else:
            m, pzts = gen_MUSE_dmg(**kw)
        beam = Beam_from_pzt(cls.N_RAYS, pzts[0], power=cls.N_RAYS / len(pzts),
                             f=350.e3, npeaks=3, nfft=cls.NFFT, t=cls.T)
        m.set_init_beam(beam)
        m.calc_t()
        m.calc_signal()
        sig = {p.name: (np.zeros(len(cls.T)) if p.signal_s is None else p.signal_s)
               for p in pzts[1:]}
        m.close_h5()
        return m, sig

    def test_transparent_damage_matches_intact(self):
        from plate_config import TH
        inv = dict(rdmg=0., bldmg=0., rmdmg=1., thdmg=TH, xidmg=1.e-3)
        _, s_int = self._run(intact=True)
        for shape in ('rect', 'ellipse'):
            m, s = self._run(shape=shape, phidmg=0.4, mesh='holes', **inv)
            self.assertEqual(len(m.mediums), 2)
            n_checked = 0
            for name in s_int:
                if not np.abs(s_int[name]).max() > 0.:
                    continue
                n_checked += 1
                self.assertLess(_nrmse(s[name], s_int[name]), 1e-6, (shape, name))
            self.assertGreaterEqual(n_checked, 5)

    def test_holes_match_cells_physical_damage(self):
        for shape in ('rect', 'ellipse'):
            m_c, s_c = self._run(shape=shape, phidmg=0.4, mesh='cells')
            m_h, s_h = self._run(shape=shape, phidmg=0.4, mesh='holes')
            self.assertEqual(len(m_h.mediums), 2)
            self.assertLess(len(m_h.rays_h), len(m_c.rays_h))  # no invisible-wall copies
            n_checked = 0
            for name in s_c:
                self.assertTrue(np.all(np.isfinite(s_h[name])))
                if not np.abs(s_c[name]).max() > 0.:
                    continue
                n_checked += 1
                self.assertLess(_nrmse(s_h[name], s_c[name]), 1e-5, (shape, name))
            self.assertGreaterEqual(n_checked, 5)

    def test_sensor_overlapping_damage_is_rejected(self):
        from MUSE_dmg import gen_MUSE_dmg
        from plate_config import PZT_POS
        x, y = PZT_POS[0]
        with self.assertRaises(ValueError):  # wall cuts the sensor circumference
            gen_MUSE_dmg(shape='rect', xdmg=x + 20., ydmg=y, xldmg=44., yldmg=20.)
        with self.assertRaises(ValueError):  # sensor entirely inside the damage
            gen_MUSE_dmg(shape='rect', xdmg=x, ydmg=y, xldmg=60., yldmg=60.)

    def test_sensor_inside_damage_hole_goes_to_inner_medium(self):
        # Map2D.add_sensor must pick the smallest bounding box that contains the
        # sensor, so a sensor placed inside the damage is registered there.
        from MUSE_dmg import gen_MUSE_dmg
        from RayTracing.Sensors import Sensor
        m, _ = gen_MUSE_dmg(shape='rect', xdmg=200., ydmg=200., xldmg=60., yldmg=60.)
        s = Sensor('circ', [np.array([200., 200.]), 3.], name='inner')
        m.add_sensor(s)
        dmg = [med for med in m.mediums.values() if len(med.objs) == 4][0]
        self.assertIn(s, dmg.sensors)


if __name__ == '__main__':
    unittest.main()

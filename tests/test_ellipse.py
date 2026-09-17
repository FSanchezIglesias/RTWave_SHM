"""Tests of the elliptical boundary (``geom.objects_2d.Ellipse``) and of the
elliptical damage layout of ``example/MUSE_dmg.py``.

Run from the repository root::

    python tests/test_ellipse.py
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

from geom.geom_utils import ellipse_seg_intersect_2d, circunf_seg_intersect_2d  # noqa: E402
from geom.objects_2d import Ellipse  # noqa: E402


def _nrmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2))


class TestEllipseKernel(unittest.TestCase):

    def test_axis_aligned_hit_and_normal(self):
        res = ellipse_seg_intersect_2d([0., 0.], 2., 1., 0., [-5., 0.], [5., 0.])
        np.testing.assert_allclose(res[:2], [-2., 0.], atol=1e-12)
        np.testing.assert_allclose(res[2:4], [-1., 0.], atol=1e-12)
        self.assertEqual(res[4], 0.)  # start point outside

    def test_rotation_swaps_axes(self):
        res = ellipse_seg_intersect_2d([0., 0.], 2., 1., np.pi / 2, [-5., 0.], [5., 0.])
        np.testing.assert_allclose(res[:2], [-1., 0.], atol=1e-12)
        res = ellipse_seg_intersect_2d([0., 0.], 2., 1., np.pi / 2, [0., -5.], [0., 5.])
        np.testing.assert_allclose(res[:2], [0., -2.], atol=1e-12)

    def test_rotated_and_translated(self):
        c = np.array([10., -3.])
        phi = 0.7
        a, b = 3., 1.5
        # a point on the ellipse at local angle psi, and the outward normal there
        psi = 1.1
        loc = np.array([a * np.cos(psi), b * np.sin(psi)])
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        p_on = c + R.dot(loc)
        n_exp = R.dot(np.array([np.cos(psi) / a, np.sin(psi) / b]))
        n_exp /= np.linalg.norm(n_exp)
        # segment along the normal, starting outside and ending inside
        res = ellipse_seg_intersect_2d(c, a, b, phi, p_on + 4 * n_exp, p_on - 0.5 * n_exp)
        np.testing.assert_allclose(res[:2], p_on, atol=1e-10)
        np.testing.assert_allclose(res[2:4], n_exp, atol=1e-10)
        self.assertEqual(res[4], 0.)

    def test_miss(self):
        self.assertIsNone(ellipse_seg_intersect_2d([0., 0.], 2., 1., 0., [-5., 3.], [5., 3.]))
        # segment ending before the ellipse
        self.assertIsNone(ellipse_seg_intersect_2d([0., 0.], 2., 1., 0., [-5., 0.], [-3., 0.]))

    def test_first_crossing_is_nearest(self):
        # crosses twice: the returned hit is the first one along the segment
        res = ellipse_seg_intersect_2d([0., 0.], 2., 1., 0., [5., 0.], [-5., 0.])
        np.testing.assert_allclose(res[:2], [2., 0.], atol=1e-12)

    def test_start_inside_gives_exit_and_inside_flag(self):
        res = ellipse_seg_intersect_2d([0., 0.], 2., 1., 0., [0., 0.], [5., 0.])
        np.testing.assert_allclose(res[:2], [2., 0.], atol=1e-12)
        np.testing.assert_allclose(res[2:4], [1., 0.], atol=1e-12)
        self.assertEqual(res[4], 1.)

    def test_nudged_spawn_point_does_not_rehit(self):
        # A reflected ray is spawned 1e-8 mm off the wall along its new
        # direction (ray_refl); it must not re-hit the curve at its start.
        hit = np.array([-2., 0.])
        r = np.array([-1., 0.3])
        r /= np.linalg.norm(r)
        start = hit + 1e-8 * r
        self.assertIsNone(ellipse_seg_intersect_2d([0., 0.], 2., 1., 0., start, start + 50 * r))
        # same for a refracted ray spawned just inside: only the far exit is found
        r_in = np.array([1., 0.3])
        r_in /= np.linalg.norm(r_in)
        start = hit + 1e-8 * r_in
        res = ellipse_seg_intersect_2d([0., 0.], 2., 1., 0., start, start + 50 * r_in)
        self.assertGreater(np.linalg.norm(res[:2] - hit), 1.)
        self.assertEqual(res[4], 1.)

    def test_circle_consistency(self):
        c, r = np.array([3., 4.]), 2.5
        p1, p2 = np.array([-4., 1.]), np.array([9., 7.])
        res = ellipse_seg_intersect_2d(c, r, r, 0.4, p1, p2)
        pts = circunf_seg_intersect_2d(c, r, p1, p2)
        nearest = min(pts, key=lambda p: np.linalg.norm(p - p1))
        np.testing.assert_allclose(res[:2], nearest, atol=1e-10)


class TestEllipseObject(unittest.TestCase):

    def test_limits_and_contains(self):
        e = Ellipse([1., 1.], 3., 1., phi=np.pi / 2)
        xmax, xmin, ymax, ymin = e.get_limits()
        np.testing.assert_allclose([xmax, xmin, ymax, ymin], [2., 0., 4., -2.], atol=1e-12)
        self.assertTrue(e.contains([1., 3.5]))
        self.assertFalse(e.contains([2.5, 1.]))


class TestMuseEllipseMesh(unittest.TestCase):
    """End-to-end checks of gen_MUSE_dmg(shape='ellipse') with a reduced beam."""

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

    def test_transparent_ellipse_matches_intact_and_rect(self):
        # Fully transparent damage (no reflection, no losses, same thickness
        # and damping as the plate): the 6-medium ellipse mesh, the 5-medium
        # rectangle mesh and the intact plate must give the same signals.
        from plate_config import TH
        inv = dict(rdmg=0., bldmg=0., rmdmg=1., thdmg=TH, xidmg=1.e-3)
        _, s_int = self._run(intact=True)
        _, s_rect = self._run(shape='rect', **inv)
        m_ell, s_ell = self._run(shape='ellipse', phidmg=0.4, **inv)
        self.assertEqual(len(m_ell.mediums), 2)
        n_checked = 0
        for name in s_int:
            if not np.abs(s_int[name]).max() > 0.:
                continue  # not reached within the short time vector
            n_checked += 1
            self.assertLess(_nrmse(s_rect[name], s_int[name]), 1e-6, name)
            self.assertLess(_nrmse(s_ell[name], s_int[name]), 1e-6, name)
        self.assertGreaterEqual(n_checked, 5)

    def test_physical_ellipse_differs_from_intact(self):
        _, s_int = self._run(intact=True)
        _, s_ell = self._run(shape='ellipse', phidmg=0.4)
        diff = 0.
        n_nonzero = 0
        for name in s_int:
            self.assertTrue(np.all(np.isfinite(s_ell[name])))
            if np.abs(s_ell[name]).max() > 0.:
                n_nonzero += 1
            if np.abs(s_int[name]).max() > 0.:
                diff = max(diff, _nrmse(s_ell[name], s_int[name]))
        self.assertGreaterEqual(n_nonzero, 5)
        self.assertGreater(diff, 1e-3)

    def test_sensor_overlapping_ellipse_is_rejected(self):
        from MUSE_dmg import gen_MUSE_dmg
        from plate_config import PZT_POS
        x, y = PZT_POS[0]
        # ellipse edge cuts the sensor circumference
        with self.assertRaises(ValueError):
            gen_MUSE_dmg(shape='ellipse', xdmg=x + 20., ydmg=y, xldmg=44., yldmg=20.)
        # sensor entirely inside the ellipse
        with self.assertRaises(ValueError):
            gen_MUSE_dmg(shape='ellipse', xdmg=x, ydmg=y, xldmg=60., yldmg=60.)
        # legacy cell mesh: sensor inside the transparent cell
        with self.assertRaises(ValueError):
            gen_MUSE_dmg(shape='ellipse', xdmg=x + 20., ydmg=y, xldmg=60., yldmg=20.,
                         mesh='cells')


if __name__ == '__main__':
    unittest.main()

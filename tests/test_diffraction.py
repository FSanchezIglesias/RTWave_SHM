"""Tests of the edge (corner) diffraction of the ray tracer.

* ``TestUTD``: the transition function and the exact half-plane solution.
* ``TestHalfPlane``: a point source in a plate with absorbing edges and a
  rigid screen ending at a free edge.  The ratio screen / no-screen of the
  350 kHz component of the signals of sensors on an arc around the edge is
  compared with the exact Sommerfeld solution.
* ``TestMuseCorner``: on the MUSE square damage the sensor amplitude must
  vary smoothly across the corner shadow line (it jumped 2.5x over 2 degrees
  without diffraction), diffracted rays are only spawned at the 4 damage
  corners, and the intact plate is unchanged.

Run from the repository root::

    python tests/test_diffraction.py
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

import RayTracing.Ray as ray_mod  # noqa: E402
from RayTracing.Ray import Beam  # noqa: E402
from RayTracing.Sensors import Sensor  # noqa: E402
from geom.objects_2d import Segment, medium, build_vertices  # noqa: E402
from geom.map_2d import Map2D  # noqa: E402
from utils_rays.utd import transition_T, sommerfeld_half_plane  # noqa: E402


class _ConstWS:
    """Isotropic, non-dispersive wave-speed model [m/s]."""

    def __init__(self, v):
        self._v = float(v)

    def S0(self, fa):
        return self._v

    def A0(self, fa):
        return self._v


def _component(sig, t, f):
    """Complex amplitude of the frequency ``f`` in ``sig(t)``."""
    return np.sum(sig * np.exp(-2j * np.pi * f * t))


class TestUTD(unittest.TestCase):

    def test_transition_function(self):
        self.assertAlmostEqual(float(transition_T(0.)), 1., places=12)
        w = np.array([5., 10., 20.])
        np.testing.assert_allclose(transition_T(w), 1. / (np.sqrt(np.pi) * w), rtol=0.05)
        self.assertTrue(np.all(np.diff(transition_T(np.linspace(0., 5., 200))) < 0.))

    def test_half_plane_limits(self):
        k, rho = 0.5, 200.
        # deep in the lit region the field is the incident one (plus a weak
        # diffracted term), on the shadow boundary it is 1/2, deep in the
        # shadow it vanishes
        u = np.abs(sommerfeld_half_plane(k, rho, np.array([np.pi, 1.5 * np.pi, 1.95 * np.pi]), 0.5 * np.pi))
        self.assertAlmostEqual(u[0], 1., delta=0.05)
        self.assertAlmostEqual(u[1], 0.5, delta=0.03)
        self.assertLess(u[2], 0.1)


class TestHalfPlane(unittest.TestCase):
    """Rigid screen (ratio_rfl=1) with a free edge inside an absorbing plate."""

    V = 5000.e3        # mm/s (5 km/s)
    F = 350.e3         # Hz
    L = 600.           # plate side [mm]
    EDGE = np.array([300., 300.])
    PHI0 = 0.75 * np.pi            # source direction seen from the edge (see _pos)
    X_SRC = 150.                   # source distance from the edge [mm]
    N_RAYS = 3001
    NFFT = 200
    T = np.linspace(0., 2.e-4, 6000)
    RHO = 120.                     # arc radius around the edge [mm]

    # Sommerfeld convention: the screen is the direction phi = 0 (here +y,
    # from the edge up to the plate boundary), phi grows counter-clockwise.
    PHIS = np.deg2rad(np.array([20., 30., 40., 45., 50., 60., 80., 100., 135., 180.,
                                225., 270., 290., 300., 310., 315., 320., 330., 340.]))

    @classmethod
    def _pos(cls, phi, r):
        return cls.EDGE + r * np.array([-np.sin(phi), np.cos(phi)])

    @classmethod
    def _run(cls, screen):
        kw_abs = dict(boundary_losses=1., ratio_rfl=1., ratio_mode=1.)   # absorbing edges
        L = cls.L
        P = [np.array([0., 0.]), np.array([L, 0.]), np.array([L, L]), np.array([0., L])]
        objs = [Segment(P[i], P[(i + 1) % 4], **kw_abs) for i in range(4)]
        if screen:
            # the screen runs from the top edge down to the free edge (x = 300, y >= 300)
            objs.append(Segment(np.array([300., L]), cls.EDGE.copy(),
                                boundary_losses=0., ratio_rfl=1., ratio_mode=1.))
        m = medium(_ConstWS(cls.V / 1.e3), 1., xi=0., dispersive=False)
        m.add_objs(objs)
        r_map = Map2D(mediums=[m])
        # sensors on an arc around the edge (the source, at PHI0 = 135 deg, lights
        # the left face; incident shadow boundary at 315 deg, reflection
        # boundary at 45 deg)
        phis = cls.PHIS
        sens = []
        for ph in phis:
            s = Sensor('circ', [cls._pos(ph, cls.RHO), 3.], name='phi%.0f' % np.degrees(ph))
            r_map.add_sensor(s)
            sens.append(s)
        beam = Beam(cls.N_RAYS, [cls._pos(cls.PHI0, cls.X_SRC)], m, f=cls.F, npeaks=3,
                    nfft=cls.NFFT, t=cls.T, kind='S0', power=cls.N_RAYS)
        r_map.set_init_beam(beam)
        r_map.calc_t()
        out = {}
        for s in sens:
            s.signal()
            out[s.name] = np.zeros(len(cls.T)) if s.signal_s is None else s.signal_s
        return r_map, phis, out

    def test_vertex_is_a_free_edge(self):
        r_map, _, _ = self._run(screen=True)
        self.assertEqual(len(r_map.vertices), 1)
        v = r_map.vertices[0]
        np.testing.assert_allclose(v.p, self.EDGE)
        self.assertEqual(len(v.faces), 1)
        self.assertEqual(len(v.walls), 1)

    def test_matches_sommerfeld(self):
        self.assertTrue(ray_mod.diffraction)
        _, phis, ref = self._run(screen=False)
        r_map, _, sig = self._run(screen=True)
        self.assertGreater(sum(1 for h in r_map.rays_h if r_map.get_ray(h).diff_order == 1), 10)

        k = 2. * np.pi * self.F / self.V
        # exact plane-wave solution with the point-source distance parameter
        # L = rho X / (rho + X) in the Fresnel arguments
        Lp = self.RHO * self.X_SRC / (self.RHO + self.X_SRC)
        exact = np.abs(sommerfeld_half_plane(k, self.RHO, phis, self.PHI0, rigid=True, L=Lp))

        model = []
        for ph, name in zip(phis, sig):
            c_ref = _component(ref[name], self.T, self.F)
            c_sig = _component(sig[name], self.T, self.F)
            model.append(abs(c_sig) / abs(c_ref))
        model = np.array(model)
        # Next to the lit face (phi < 60 deg) the incident and reflected waves
        # form standing-wave fringes that 3 mm sensors sampling discrete rays
        # cannot resolve point by point; compare the rest of the arc, which
        # crosses the incident shadow boundary at 315 deg.
        sel = np.degrees(phis) >= 80.
        rms = np.sqrt(np.mean((model[sel] - exact[sel]) ** 2))
        self.assertLess(rms, 0.12, (model.round(3), exact.round(3)))
        # half the incident field on the shadow boundary, shadow filled with
        # the exact decay (each deep-shadow probe within 0.12 of the exact
        # value and above half of it; pure GO gives 0 there)
        i_sb = int(np.argmin(np.abs(np.degrees(phis) - 315.)))
        self.assertAlmostEqual(model[i_sb], 0.5, delta=0.12)
        for deg in (320., 330., 340.):
            i = int(np.argmin(np.abs(np.degrees(phis) - deg)))
            self.assertAlmostEqual(model[i], exact[i], delta=0.12, msg=deg)
            self.assertGreater(model[i], 0.5 * exact[i], deg)


class TestTotalInternalReflection(unittest.TestCase):
    """Beyond the critical angle the transmitted share is reflected, not lost."""

    def _run(self, tir):
        from RayTracing.Ray import Ray
        from scipy.fft import rfftfreq
        old = ray_mod.total_internal_reflection
        ray_mod.total_internal_reflection = tir
        try:
            kw = dict(boundary_losses=0., ratio_rfl=0.1, ratio_mode=1.)
            wall = Segment(np.array([50., 0.]), np.array([50., 100.]), **kw)
            m_slow = medium(_ConstWS(1000.), 1., xi=1.e-12, dispersive=False)   # 1e6 mm/s
            m_fast = medium(_ConstWS(2000.), 1., xi=2.e-12, dispersive=False)
            m_slow.add_objs([wall]); m_fast.add_objs([wall])
            r_map = Map2D(mediums=[m_slow, m_fast])
            t = np.linspace(0., 1.e-4, 200)
            freq = np.ones(33, dtype=complex)
            fft_freq = rfftfreq(64, d=t[1] - t[0])
            # 45 deg incidence from the slow side: sin_t = 2 * sin 45 > 1
            ray = Ray(np.array([0., 0.]), np.array([1., 1.]), freq=freq, medium=m_slow,
                      t=t, t0=0., kind='S0', a=1., _fft_freq=fft_freq)
            r_map.save_ray(ray); r_map.rays_h = [ray.__hash__()]
            r_map.calc_t(t=1.e-4)
            return ray, r_map
        finally:
            ray_mod.total_internal_reflection = old

    def test_transmitted_share_is_reflected(self):
        ray, r_map = self._run(tir=True)
        self.assertEqual(len(r_map.rays_h), 1)              # no refracted ray
        self.assertLess(ray.d[1][0], 0.)                    # reflected
        self.assertAlmostEqual(ray.a[1], 1., places=9)      # (1 - bl) = 1, nothing lost
        ray0, r_map0 = self._run(tir=False)
        self.assertAlmostEqual(ray0.a[1], 0.1, places=9)    # v1: ratio_rfl only


class TestMuseCorner(unittest.TestCase):
    """Continuity across the corner shadow line of the MUSE square damage."""

    N_RAYS = 2001
    NFFT = 300
    T = np.linspace(0., 1.e-4, 5000)
    SRC = np.array([181.5, 145.2])
    V1 = np.array([331., 171.])     # top-left corner of the 48 mm square at (355, 147)
    R = 270.
    DAS = np.array([-4., -2., -1., 1., 2., 4.])

    @classmethod
    def _run(cls, damaged):
        from MUSE_dmg import gen_MUSE_dmg, gen_MUSE_intact, ws
        from RayTracing.Ray import Beam_from_pzt
        if damaged:
            m, pzts = gen_MUSE_dmg(xdmg=355., ydmg=147., xldmg=48., yldmg=48., thdmg=3.,
                                   rdmg=0.1, bldmg=0.05, rmdmg=1, shape='rect')
        else:
            m, pzts = gen_MUSE_intact()
        ang_c = np.arctan2(cls.V1[1] - cls.SRC[1], cls.V1[0] - cls.SRC[0])
        sens = {}
        for da in cls.DAS:
            a = ang_c + np.deg2rad(da)
            p = cls.SRC + cls.R * np.array([np.cos(a), np.sin(a)])
            sens[da] = Sensor('circ', [p, 4.], name='c%+.0f' % da)
            m.add_sensor(sens[da])
        beam = Beam_from_pzt(cls.N_RAYS, pzts[0], power=cls.N_RAYS / 8, f=350.e3, npeaks=3,
                             nfft=cls.NFFT, t=cls.T, kind='S0')
        m.set_init_beam(beam)
        m.calc_t()
        v = ws.S0((350.e3 * 1.288 / 1e6, 0.)) * 1e3
        win = cls.T < (cls.R + 60.) / v          # direct arrival only
        out = {}
        for da, s in sens.items():
            s.signal()
            out[da] = np.abs(s.signal_s[win]).max()
        return m, out

    def test_corner_transition_is_smooth(self):
        self.assertTrue(ray_mod.diffraction)
        _, ref = self._run(damaged=False)
        m, sig = self._run(damaged=True)
        ratios = np.array([sig[da] / ref[da] for da in self.DAS])
        # diffracted rays only at the 4 damage corners
        vs = [v for med in m.mediums.values() for v in med.vertices]
        self.assertEqual(len(m.vertices), 4)
        self.assertEqual(len(set(id(v) for v in vs)), 4)
        n_fans = sum(1 for h in m.rays_h if m.get_ray(h).diff_order == 1)
        self.assertGreater(n_fans, 100)
        # smooth: monotone across the line, no factor > 1.4 between probes
        # 1-2 degrees apart (2.5 without diffraction), ~1/2 next to the line
        self.assertTrue(np.all(np.diff(ratios) > -0.05), ratios)
        self.assertLess(np.max(ratios[1:] / ratios[:-1]), 1.4, ratios)
        self.assertGreater(ratios[2], 0.3, ratios)
        self.assertLess(ratios[3], 0.75, ratios)

    def test_intact_plate_has_no_vertices(self):
        from MUSE_dmg import gen_MUSE_intact
        m, _ = gen_MUSE_intact()
        self.assertEqual(len(m.vertices), 0)


if __name__ == '__main__':
    unittest.main()

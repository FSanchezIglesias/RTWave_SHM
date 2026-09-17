"""Check that ``Sensor._signal_on_ray`` (step recurrence, per-run segment
lookup) reproduces the original per-point integration semantics of
``Ray.signal_at_x``:

* every integration point uses the ray event (segment) it lies in, even when a
  chord spans a reflection point;
* points at or beyond the ray's last recorded position are skipped;
* the dispersion coefficient is the one of the segment's own direction
  (``Ray.phase_coeff_at``), both with ``dispersion_follows_direction`` True and
  False.

Run with ``python tests/test_signal_on_ray.py`` or ``pytest tests``.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, 'example')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from scipy.fft import irfft  # noqa: E402

import RayTracing.Ray as ray_module  # noqa: E402
from RayTracing.Ray import Beam  # noqa: E402
from RayTracing.Sensors import Sensor  # noqa: E402
from geom.objects_2d import Segment, medium  # noqa: E402
from geom.map_2d import Map2D  # noqa: E402
from wavespeed import wavespeed_composite  # noqa: E402
from plate_config import WAVESPEED_HDF5, TH  # noqa: E402


def _reference_signal(ray, xs_ray: list, d_x: float, size: float) -> np.ndarray:
    """Per-point reference: v1 ``Ray.signal_at_x`` semantics with the
    dispersion coefficient of the segment's own direction."""
    from scipy.signal.windows import hamming

    xs_ray = sorted(set(xs_ray))
    t = ray.t
    total = np.zeros(len(t))
    for i in range(len(xs_ray) // 2):
        xi = np.arange(xs_ray[2 * i], xs_ray[2 * i + 1], d_x) + d_x / 2
        w = hamming(xi.size) * abs(xs_ray[2 * i + 1] - xs_ray[2 * i]) / size
        for k, x in enumerate(xi):
            try:
                seg = next(j for j, v in enumerate(ray.x) if v > x) - 1
            except StopIteration:
                continue
            x0, f0, a0, t0 = ray.x[seg], ray.freq[seg], ray.a[seg], ray.int_times[seg]
            v = ray.medium.v_ray(ray, seg)
            x_i = x - x0
            t_i = t0 + x_i / v
            f_i = np.exp(ray.phase_coeff_at(seg) * x_i) * f0
            a_i = a0 * np.exp(-2 * np.pi * ray._dom_freq * ray.medium.xi * (t_i - t0))
            s = a_i * irfft(f_i, n=len(t))
            s[t < t_i / 2] = 0
            total += s * d_x * w[k]
    return total


def _reflected_ray(follow: bool):
    """Trace one S0 ray that reflects once off the right plate edge."""
    ray_module.dispersion_follows_direction = follow
    ws = wavespeed_composite(WAVESPEED_HDF5)
    m = medium(ws, TH, xi=1.e-3)
    L = 100.
    P = [np.array([0., 0.]), np.array([L, 0.]), np.array([L, L]), np.array([0., L])]
    m.add_objs([Segment(P[k], P[(k + 1) % 4], boundary_losses=0.25, ratio_mode=1.)
                for k in range(4)])
    t = np.linspace(0., 4.e-5, 2000)
    # one ray from (20, 30) at 20 deg: hits x = L after ~85 mm and reflects
    beam = Beam(1, [np.array([20., 30.]), np.deg2rad(20.), np.deg2rad(20.)], m,
                kind='S0', f=350.e3, npeaks=3, nfft=100, t=t)
    mp = Map2D(mediums=[m], background=True)
    mp.set_init_beam(beam)
    mp.calc_t()
    ray = mp.get_ray(beam.rays[0].__hash__())
    assert len(ray.x) >= 3, ray.x
    assert not np.allclose(ray.d[0], ray.d[-1]), 'ray did not reflect'
    return ray


def _check(follow: bool) -> None:
    ray = _reflected_ray(follow)
    x_refl = ray.x[1]
    x_end = ray.x[-1]
    sens = Sensor('circ', [[0., 0.], 4.], name='probe')
    d_x = 0.1
    chords = [
        [x_refl - 5., x_refl + 5.],                # spans the reflection event
        [x_end - 3., x_end + 3.],                  # runs past the ray end
        [x_refl - 5., x_refl + 5., x_end - 3., x_end + 3.],  # both, as one xs_ray
    ]
    for xs in chords:
        ref = _reference_signal(ray, xs, d_x, sens.size)
        new = sens._signal_on_ray(ray, xs, d_x, window='hsphere')
        scale = np.abs(ref).max()
        assert scale > 0, 'reference signal is identically zero'
        err = np.abs(new - ref).max() / scale
        assert err < 1.e-10, (follow, xs, err)
    if follow:
        # the two segments must really use different coefficients
        assert not np.allclose(ray.phase_coeff_at(0), ray.phase_coeff_at(1))
    else:
        assert np.array_equal(ray.phase_coeff_at(0), ray.phase_coeff_at(1))


def test_signal_on_ray_follows_direction() -> None:
    _check(True)


def test_signal_on_ray_birth_direction() -> None:
    _check(False)


if __name__ == '__main__':
    test_signal_on_ray_follows_direction()
    test_signal_on_ray_birth_direction()
    print('OK')

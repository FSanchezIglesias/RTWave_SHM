# RTWave_SHM
Code for simulation of elastic wave propagation on thin plates via ray tracing

## Install
To install simply clone from github

```bash
git clone https://github.com/FSanchezIglesias/RTWave_SHM.git code
```

or:
```bash
git clone git@github.com:FSanchezIglesias/RTWave_SHM.git code
```

## Simple usage example

### Import the objects to be used

```python
from geom.objects_2d import Segment, medium
from RayTracing.Sensors import Sensor
from RayTracing.Ray import Beam
from geom.map_2d import Map2D
```

### Geometry:
```python
s1 = Segment(np.array([-500, -100]), np.array([500, 100]))
pzt = Sensor('circ', [[0,-200], 12.])
```

Curved boundaries are `Ellipse(c, a, b, phi)` (centre, semi-axes, rotation
in rad; `Ellipse(c, r, r)` is a circle). They take the same
`boundary_losses`/`ratio_rfl`/`ratio_mode` parameters as `Segment` and are
added to mediums the same way. Rays interact with the *nearest* boundary
crossed by their trace, so a medium may be non-convex: a damage is simply a
hole in the plate medium (its boundary objects are added to both mediums)
and no auxiliary invisible walls are needed.

The MUSE example supports both damage shapes:
`gen_MUSE_dmg(shape='rect')` (default, `XLDMG x YLDMG` box) and
`gen_MUSE_dmg(shape='ellipse', phidmg=<rad>)` (ellipse with full axes
`XLDMG`, `YLDMG`); `example/plate_config.py` holds `DMG_SHAPE` / `PHIDMG`.
Both build 2 mediums (plate with the damage as a hole + damage). The
legacy convex-cell mesh with invisible walls is still available as
`gen_MUSE_dmg(..., mesh='cells')`; it is only needed for solvers that
interact with the first rather than the nearest crossed wall (v1).

### Define a wave speed class
```python
class WS:
    def __init__(self, E, nu, rho):
        # Assuming limit propagation speeds as an example
        # Lambda functions to ignore dependency of freq and angle
        self.S0 = lambda x: np.sqrt((E * (1 - nu)) / (rho * (1 + nu) * (1 - 2 * nu))) /1000
        self.A0 = lambda x: np.sqrt(E / (2 * rho * (1 + nu))) /1000
```

### Create 2 mediums separated by s1
Medium 1: aluminium
```python
m1 = medium(WS(70000., 0.31, 2.7E-9), 1.)
m1.add_objs([s1,])
```
Medium 2: titanium
```python
m2 = medium(WS(110000., 0.27, 4.5E-9), 2., xi=1.e-3)
m2.add_objs([s1, pzt])
```

### Initial rays
Time vector to study:
```python
import numpy as np
t = np.linspace(0., 0.0001, 100000)
```
Initial ray beam definition of 40 rays (20 symmetric and 20 antisymmetric)
```python
nrays = 20
ibeam = Beam(nrays, [[0,200],],
             medium=m1,
             f=300.e+3, npeaks=3, nfft=500, t=t)
```

### Define the map and calculate
```python
m = Map2D(ibeam, [m1,m2])
m.calc_t()
```

### Obtain signal at sensor
```python
y = pzt.signal()

import matplotlib.pyplot as plt
plt.plot(t, y)
```

### Plot a representation of the ray map
```python
m.plot2d()
```

## Versions

- **v1** — original pure-Python/NumPy reference implementation (commit `9b00afb`).
- **v2** — same geometry and public API, internals rewritten for speed (Numba
  kernels, in-memory ray store, cached per-ray invariants, step-recurrence
  chord integration). Two deliberate model changes relative to v1:
  - the excitation burst has unit amplitude, so signals scale linearly with the
    per-ray power `a0 = power / (2 * n_rays)` (v1 scaled with `a0**2`);
  - after every reflection the dispersion curve is re-evaluated for the new
    propagation direction (`RayTracing.Ray.dispersion_follows_direction`,
    default `True`; set it to `False` to recover the v1 behaviour, which kept
    the birth-direction curve for the whole life of a ray).

## Edge diffraction

Geometrical optics casts a sharp shadow behind every corner: two rays a
hair apart at a vertex of the damage take completely different paths, and the
sensor amplitude jumped by 2.5x within 2 degrees across the corner shadow
line of the MUSE square (a physical shadow needs a Fresnel number
`a^2 / (lambda * distance) >> 1`; for the 48 mm square and the 14.9 mm S0
wavelength it is 0.4 at 100 mm behind the damage). The solver therefore
adds edge diffraction in the spirit of the Uniform Theory of Diffraction
(Keller; Kouyoumjian & Pathak): every convex corner of a wall is a secondary
source.

- `geom.objects_2d.build_vertices` (called by `Map2D`) finds the `Vertex`
  objects: endpoints shared by two `Segment` walls, or free edges (the tip of
  a screen). The outline of a medium (the plate) does not diffract; corners
  shared by 3+ walls (the legacy cell mesh) are skipped with a warning.
- In `Ray.trace`, the one ray of each ray family that passes within half a
  ray spacing of a vertex (`dtheta * (x_src + x)`, so exactly one ray per
  family) calls `utils_rays.ray_utils.ray_diff`. Two probe rays straddling the
  vertex are traced through the walls of the obstacle, carrying the ray's
  spectrum dispersed along every leg; every outgoing family (incident,
  reflected, transmitted, entered-next-to-the-corner-and-left-through-the-
  adjacent-face, ...) whose spectrum differs between the probes is a shadow
  boundary whose jump is the complex difference spectrum (so a transmitted
  wave that is delayed and chirped by the thicker region is compensated
  with its actual waveform, not just an amplitude ratio). A transparent
  wall gives no fan.
- For each boundary a fan of rays is launched from the vertex: spectrum
  half the jump, negative where the family exists and positive in its
  shadow, amplitude that of the incident ray (times the fan/incident spacing
  ratio), and a per-ray amplitude law `Ray.amp_factor(x) =
  rho/(X+rho) * T(w)` with `T` the Fresnel transition function
  (`utils_rays.utd`) evaluated at the dominant frequency and `X` the path
  from the family origin to the vertex, so that the fan (ray density
  `1/rho`) decays like the incident family and GO + fan is continuous.
  Fans are dense (2x the incident spacing) within 3 degrees of the boundary
  and sparser further out, up to 90 degrees; the amplitude of a ray is
  proportional to its angular coverage.
- Module switch `RayTracing.Ray.diffraction` (default `True`) and parameters
  `RayTracing.Ray.diff_params`. Sensors apply the amplitude law in
  `Sensor._integrate_run`; the video renderer (`example/plot_wave_video.py`)
  spreads every ray over its tube width so that sparse fan rays do not show
  as streaks.
- Validation (`tests/test_diffraction.py`): a rigid screen with a free edge
  in an absorbing plate reproduces the exact Sommerfeld half-plane solution
  (350 kHz component, screen / no screen): 0.59 on the shadow boundary
  (exact 0.46-0.53), 0.42 / 0.30 / 0.20 at 5 / 15 / 25 degrees into the
  shadow (exact 0.37 / 0.19 / 0.19), 0.73 on the reflection boundary
  (exact 0.68). On the MUSE 48 mm square the ratio to the intact plate across
  the top-left corner line at 270 mm from the source goes 0.28, 0.42, 0.46,
  0.47 | 0.50, 0.56, 0.59, 0.66, 0.77 at -6, -4, -2, -1 | +1, +2, +4, +6
  degrees (pure GO: 0.35, 0.34, 0.08, 0.00 | 1.00, 1.00, 1.00, 1.00).
- Independent reference (`tests/compare_fdtd_corner.py`): a 2D
  finite-difference solution of the scalar wave equation for the same
  idealised problem (isotropic, non-dispersive plate 5.205 mm/us with the
  48 mm square at 4.432 mm/us, absorbing edges, point source). Across the
  corner shadow line the ray model with diffraction deviates from the
  reference by 0.13 RMS (0.37 without); on receiver lines behind the damage
  by 0.38 (0.48 without), the remainder being the sharper ray caustics of
  the focusing inside the slower square.
- That comparison also exposed a lost-energy bug at **total internal
  reflection**: when Snell's law has no solution the solver dropped the
  transmitted share, so the slower square could not act as the light pipe
  the reference shows (2x on-axis amplitude). The wall now reflects
  `(1 - bl)` in that case (`RayTracing.Ray.total_internal_reflection`,
  default `True`; the v1 harness runs with `--v1-physics`, which turns both
  it and the diffraction off). The probe tracing of `ray_diff` applies the
  same rule.
- Not modelled: the tangent shadow boundary of the ellipse (creeping
  waves), mode conversion at the edge, the frequency dependence of the
  transition (dominant frequency only), diffraction of diffracted rays
  (`max_order=1`). Cost on the full benchmark (12 mm square, one source):
  17.3k rays and 3.4 s against 4.8k rays and 1.0 s with the v1 physics (no diffraction, no total internal reflection). The
  regression harnesses `tests/compare_v1_v2.py` and
  `tests/compare_cells_holes.py` run with `--no-diffraction` (v1 and the cell
  mesh have no diffracting vertices).

## Performance

Benchmark: one source, 2001 rays, 726 × 726 mm CFRP plate with a 12 × 12 mm
damage region, 10 000 samples at 50 MHz, 500 FFT bins per ray. NRMSE is the
RMSE over all receiving sensors and samples divided by the RMS of the
reference signals (1.63e-2); values around 1e-14 are double-precision
round-off.

| Step | Optimization | Time (s) | Incr. (×) | Cumul. (×) | NRMSE |
|---:|---|---:|---:|---:|---:|
| 0 | Original (pure Python) | 2525.09 | — | 1.0 | — |
| 1 | Numba compilation | 231.36 | 10.91 | 10.9 | 1.1e-14 |
| 2 | In-memory ray cache | 115.19 | 2.01 | 21.9 | 1.1e-14 |
| 3 | Zero-amplitude culling | 51.43 | 2.24 | 49.1 | 1.1e-14 |
| 4 | Hash caching | 48.62 | 1.06 | 51.9 | 1.1e-14 |
| 5 | Lazy `fft_speed` (dead rays) | 48.11 | 1.01 | 52.5 | 1.1e-14 |
| 6 | Vectorised `fft_speed` | 22.80 | 2.11 | 110.7 | 1.1e-14 |
| 7 | Signal integration hoisting | 21.85 | 1.04 | 115.6 | 1.2e-14 |
| 8 | GC control & medium caching | 21.58 | 1.01 | 117.0 | 1.2e-14 |
| 9 | Coarser integration step † | 19.47 | 1.11 | — | 4.2e-2 |
| 10 | Cached dominant frequency | 21.50 | 1.004 | 117.4 | 1.2e-14 |
| 11 | Pre-stored phase coefficient | 21.16 | 1.02 | 119.4 | 1.2e-14 |
| 12 | Batched 2-D interpolation | 10.16 | 2.08 | 248.6 | 1.3e-14 |
| 13 | Shared FFT frequency array | 8.80 | 1.15 | 286.9 | 1.3e-14 |
| 14 | Step-recurrence | 7.56 | 1.16 | 334.0 | 1.9e-13 |

† Step 9 (coarser integration step) was **reverted**; step 10 onward builds on
step 8.

End-to-end check of the final v2 against v1 (`tests/compare_v1_v2.py`, same
benchmark, both with the unit-amplitude burst): with
`dispersion_follows_direction = False` the NRMSE is 5.0e-9 (the residual comes
from the 1e-8 mm offset applied to reflected rays); with the default `True`
the reflected arrivals change and the NRMSE is 0.15, while the direct-arrival
windows of the nearest sensors agree to 2e-4 – 2e-3.

Two further exact steps were added after that validation (measured on a
different workstation, best of three runs; NRMSE vs the solver output before
the step; see `Optimization/New/Performance_Evolution_stage3.csv`):

| Step | Optimization | Tracing (s) | Signal (s) | Total (s) | Incr. (×) | NRMSE |
|---:|---|---:|---:|---:|---:|---:|
| 14 | validated solver (baseline) | 1.57 | 0.86 | 2.43 | — | — |
| 15 | one inverse FFT per chord run (cumulative spectra + direct DFT on the mask transition window) | 1.50 | 0.20 | 1.70 | 1.42 | 2.3e-16 |
| 16 | direct Numba kernel calls with pre-stored float64 geometry; byte-based ray hash | 1.26 | 0.19 | 1.45 | 1.17 | 2.3e-16 |

## Tests

```bash
python tests/test_signal_on_ray.py
python tests/test_ellipse.py
python tests/test_hole_mesh.py
python tests/test_diffraction.py
python tests/compare_fdtd_corner.py
python tests/compare_v1_v2.py --v1-dir <v1 snapshot dir> --nrays 201
python tests/compare_cells_holes.py [--full] [--shape ellipse] [--plot]
```

`tests/test_hole_mesh.py` checks the nearest-hit tracing and that the
2-medium hole mesh reproduces the legacy cell mesh (NRMSE < 1e-5) and, for
a transparent damage, the intact plate. `tests/compare_cells_holes.py` runs
the same comparison as two subprocesses (hole mesh vs cell mesh, current
solver) and prints per-sensor NRMSE, ray counts and timings: on the full
benchmark the meshes agree to 8e-9 and the hole mesh is ~1.2x faster.
`tests/test_ellipse.py` checks the ellipse intersection kernel and that a
fully transparent elliptical damage (`rdmg=0, bldmg=0, rmdmg=1, thdmg=TH,
xidmg=1e-3`) reproduces the intact-plate signals (NRMSE < 1e-6).

The v1 snapshot is `git archive 9b00afb geom RayTracing utils_rays __init__.py`
extracted into a folder, with `signal_f(self.t, self.a0, f, fd)` in
`RayTracing/Ray.py` changed to `signal_f(self.t, 1.0, f, fd)`.

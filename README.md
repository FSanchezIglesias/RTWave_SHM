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

## Tests

```bash
python tests/test_signal_on_ray.py
python tests/compare_v1_v2.py --v1-dir <v1 snapshot dir> --nrays 201
```

The v1 snapshot is `git archive 9b00afb geom RayTracing utils_rays __init__.py`
extracted into a folder, with `signal_f(self.t, self.a0, f, fd)` in
`RayTracing/Ray.py` changed to `signal_f(self.t, 1.0, f, fd)`.

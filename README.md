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

- **v1** — original, pure-Python reference implementation.
- **v2** — same physics, geometry and public API as v1, with the internals rewritten for speed (see [Performance](#performance) below). Numerical results match v1 to within machine precision, except where noted.

## Performance

Starting from the original pure-Python implementation (v1), the solver was optimized through a sequence of incremental steps, each validated against the original output (NRMSE) before being accepted. The result (v2) is **~334x** faster than v1, dropping the reference run from 2525.09 s to 7.56 s.

| Step | Optimization | Time (s) | Incr. (×) | Cumul. (×) | NRMSE |
|---:|---|---:|---:|---:|---:|
| 0 | Original (pure Python) | 2525.09 | — | 1.0 | — |
| 1 | Numba compilation | 231.36 | 10.91 | 10.9 | 1.8e-16 |
| 2 | In-memory ray cache | 115.19 | 2.01 | 21.9 | 1.8e-16 |
| 3 | Zero-amplitude culling | 51.43 | 2.24 | 49.1 | 1.8e-16 |
| 4 | Hash caching | 48.62 | 1.06 | 51.9 | 1.8e-16 |
| 5 | Lazy `fft_speed` (dead rays) | 48.11 | 1.01 | 52.5 | 1.8e-16 |
| 6 | Vectorised `fft_speed` | 22.80 | 2.11 | 110.7 | 1.8e-16 |
| 7 | Signal integration hoisting | 21.85 | 1.04 | 115.6 | 1.9e-16 |
| 8 | GC control & medium caching | 21.58 | 1.01 | 117.0 | 1.9e-16 |
| 9 | Coarser integration step † | 19.47 | 1.11 | — | 6.8e-4 |
| 10 | Cached dominant frequency | 21.50 | 1.004 | 117.4 | 1.9e-16 |
| 11 | Pre-stored phase coefficient | 21.16 | 1.02 | 119.4 | 2.0e-16 |
| 12 | Batched 2-D interpolation | 10.16 | 2.08 | 248.6 | 2.1e-16 |
| 13 | Shared FFT frequency array | 8.80 | 1.15 | 286.9 | 2.1e-16 |
| 14 | Step-recurrence | 7.56 | 1.16 | 334.0 | 3.0e-15 |

*"Incr."* is the speedup relative to the previous accepted step; *"Cumul."* is relative to the original implementation (step 0).

† Step 9 (coarser integration step) pushed the NRMSE above an acceptable threshold and was **reverted**; step 10 onward builds on step 8, not step 9.

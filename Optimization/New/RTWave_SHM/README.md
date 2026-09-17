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

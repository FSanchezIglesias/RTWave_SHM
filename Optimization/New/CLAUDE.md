# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python: `C:\Users\DdRV\venvs\PhD\Scripts\python.exe`
- pip: `C:\Users\DdRV\venvs\PhD\Scripts\pip.exe`
- Tests: `C:\Users\DdRV\venvs\PhD\Scripts\python.exe -m pytest`
- Async tests: use `anyio`, not `asyncio`

## Running Simulations

```powershell
# Run the main damage simulation (single damage position)
C:\Users\DdRV\venvs\PhD\Scripts\python.exe 02_Run-Single-Damage.py

# Results land in results/<run_name>_<freq>k_x<xdmg>.hdf5
# Logs land in logs/<run_name>_row<xdmg>.log
```

## Code Style

- Type hints required for all functions
- Public APIs must have docstrings
- PEP 8 naming: `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- f-strings for formatting, 88-char line limit
- Use early returns to avoid nesting

---

## Architecture

### Physical model

Guided Lamb Waves (GLW) propagate on a thin composite plate (AS4/8552, `(+45,-45,90,0)` layup, 726×726 mm). Damage is modeled as a rectangular zone with altered acoustic impedance. 8 circular PZT sensors are placed at fixed grid positions. Each PZT is used as a source in turn; the resulting waveforms at all other PZTs form the dataset for damage localization ML models.

### RTWave_SHM library (`RTWave_SHM/`)

The simulation proceeds in four steps: **geometry → beam → calc_t → calc_signal**.

```
Map2D                          # orchestrates everything
  └─ mediums: dict[hash → medium]
       ├─ medium               # acoustic medium (wavespeed fn, thickness, damping xi)
       │    ├─ objs: list[Segment]   # boundaries that rays can reflect/refract on
       │    └─ sensors: list[Sensor] # sensors registered in this medium
       └─ sensors: list[Sensor]      # flat list for signal calculation
```

**`geom/objects_2d.py`** — geometric primitives:
- `medium`: holds the wave-speed function, thickness, damping, and all child objects. `fshift_dispersion()` implements the FFT phase-shift model for dispersive propagation; `tl()` applies exponential amplitude decay.
- `Segment`: a reflective/refractive boundary. `intersect()` spawns reflected and refracted child rays via `ray_refl()` / `ray_refr()`.
- `Circunf`, `Polygon`, `Circle`: non-intersecting geometry (used for sensors and visualization).

**`geom/geom_utils.py`** — Numba-JIT math primitives (`dot_2d`, `norm_2d`, `cross_2d`, `seg_seg_intersect_2d`, `circunf_seg_intersect_2d`). All hot-path intersection checks call these.

**`geom/map_2d.py` — `Map2D`**:
- Primary ray store is an **in-memory dict** (`ray_cache`); HDF5 is a fallback/persistence layer only.
- `calc_t()`: iterates `rays_h` (a growing list of ray hashes), calls `ray.trace()` on each, and appends newly spawned child ray hashes. GC is disabled during this loop.
- `calc_signal()`: calls `Sensor.signal()` for each non-source sensor.
- `save_signals()`: writes sensor waveforms to HDF5 at a caller-supplied key.

**`RayTracing/Ray.py`** — `Ray` and `Beam`:
- `Ray` stores its entire propagation history as lists: `trace_points`, `d` (directions), `freq` (FFT spectrum), `a` (amplitude), `int_times`. All indexed by reflection/refraction event.
- `_phase_coeff` is pre-computed per ray (and updated after each reflection via `_update_phase_coeff()`): it encodes the anisotropic dispersion relation so that `fshift_dispersion` reduces to a pointwise complex multiply.
- `fft_speed` is computed via `batch_speed()` — a single Numba call for all 500 FFT bins at once, not 500 Python→Numba round-trips.
- `Beam` / `Beam_from_pzt`: create `nrays` rays fanning 360° from a PZT origin. Both S0 and A0 modes are spawned unless `kind` is restricted.

**`RayTracing/Sensors.py` — `Sensor`**:
- Detects ray crossings geometrically (does not alter rays). Stores crossing distances in `int_rays[ray_hash]`.
- `_signal_on_ray()`: integrates the dispersed spectrum along each ray's path through the sensor using a step-recurrence trick (advances phase and amplitude by `exp(coeff * d_x)` via multiplication, not re-evaluating exp per step).

**`RayTracing/Signal.py`** — burst signal generators (`burst_hann` is the default). Input signal is Hann-windowed at `f=350 kHz`, `npeaks=3`.

**`utils_rays/ray_utils.py`**:
- `ray_refl()`: specular reflection + optional mode-change child ray. Includes an early-return for invisible walls (`ratio_rfl=0`).
- `ray_refr()`: Snell's law refraction into a second medium.
- `save_ray()` / `load_ray()`: HDF5 serialization. `load_ray` bypasses `Ray.__init__` via `__new__` and populates slots directly.

### Wavespeed (`wavespeed.py`)

Two classes share the same `.S0(xt)` / `.A0(xt)` interface:
- `wavespeed`: isotropic material; 1D interpolation on `(freq*th, speed)` CSV data or computed Lamb dispersion curves.
- `wavespeed_composite`: anisotropic composite; 2D bilinear interpolation on `(freq*th, theta)` grid loaded from `muse_stacking.hdf5`. Exposes `batch_speed(kind, x_vals, theta)` for vectorized queries.

Both use Numba-JIT kernels (`interp1d_numba`, `interp2d_regular_numba`, `interp2d_batch_fixed_y`).

Speed units: the wavespeed functions return **km/s**; callers multiply by `1e3` to get **m/s**. Frequency argument is `freq * th / 1e6` (i.e., `f·h` in MHz·mm).

### MUSE geometry (`MUSE_dmg.py`)

`gen_MUSE_dmg()` builds a 5-medium mesh around a rectangular damage zone. It tries a **vertical** mesh first (big left/right mediums); if any invisible internal wall would cross a PZT, it falls back to a **horizontal** mesh (big top/bottom mediums). Invisible walls (`ratio_rfl=0`, `boundary_losses=0`) partition the plate without affecting the physics — they exist only to force rays into the correct medium when crossing the damage region.

`gen_MUSE_intact()` returns a single-medium plate for intact-plate baselines.

### Typical simulation flow

```python
m, pzts = gen_MUSE_dmg(xdmg=..., ydmg=..., ...)
ibeam = Beam_from_pzt(nrays, pzts[source], f=350e3, npeaks=3, nfft=500, t=t)
m.set_init_beam(ibeam)
m.calc_t()          # ray tracing (expensive)
m.calc_signal()     # integrate signal at each sensor
m.save_signals(hdf5_fname, key)
m.close_h5()
```

### HDF5 output format

Results file: `results/<run_name>_<freq>k_x<xdmg>.hdf5`
- `time`: time vector
- `<run_name>_<xdmg>_<ydmg>/PZT<n>`: sensor signal matrix `(n_time, n_sensors)`, attribute `columns` lists sensor names.
- File-level attrs: `Stacking`, `Material`, `Plate dimensions`.

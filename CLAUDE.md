# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

RTWave_SHM simulates elastic (Lamb) wave propagation on thin plates via 2D ray tracing, for structural health monitoring (SHM) applications. Rays representing S0/A0 Lamb wave modes are launched from a source, propagate through `medium` regions, reflect/refract/mode-convert at `Segment`/`Circunf` boundaries, and are recorded by `Sensor` objects to reconstruct time-domain signals (e.g. piezo/PZT sensor response).

There is no build system, package manifest (`setup.py`/`pyproject.toml`) or linter configured. The project is used by importing its modules directly (see `README.md` and `Prop_2d.py` for the intended import pattern) and running scripts such as those in `example/`. Two test scripts live in `tests/`: `test_signal_on_ray.py` (unit test of the sensor chord integration) and `compare_v1_v2.py` (end-to-end comparison of the current solver against a frozen copy of the original v1 code, see `README.md`).

The current code is "v2": the original v1 implementation (commit `9b00afb`) rewritten for speed (~300× on the benchmark in `README.md`). `Optimization/New/` holds the working folder in which that rewrite was developed (its own copy of the package under `Optimization/New/RTWave_SHM`, the thesis text `Optimization/New/optimization/Process.tex`, the per-step result files and `Performance_Evolution.csv`); the package at the repo root supersedes that copy.

## Environment

- Python, dependency-managed only via `requirements.txt` (currently just pins `numpy~=2.0.1`). In practice the code also requires `scipy`, `numba`, `h5py`, `tqdm`, `matplotlib`, and the `names` package (used by `RayTracing.Sensors.Sensor` only when a sensor is created without `name=`); `pandas` is optional (`Map2D.save_signals(use_pandas=True)`). Install these manually if missing — `requirements.txt` is not authoritative.
- Run scripts from the repository root (or with the repo root on `PYTHONPATH`) since imports are absolute, e.g. `from geom.objects_2d import Segment, medium`.
- If operating as the `DdRV` PC user, use the venv at `C:\Users\DdRV\venvs\PhD` (e.g. `C:\Users\DdRV\venvs\PhD\Scripts\python.exe`) for running/installing dependencies rather than the system Python or another venv.

## Running code

To exercise the simulation:
- Follow the walkthrough in `README.md`, or run `Prop_2d.py`-style imports.
- The `example/` directory has real, larger scenarios (`example/plate_config.py` holds the plate/PZT/damage parameters, `example/MUSE_dmg.py` builds the 5-medium damage mesh, `example/wavespeed.py` provides the Numba-compiled dispersion-table interpolators, `example/Run-Single-Damage.py` drives full runs and writes results to `results/*.hdf5` with logs under `logs/`). These scripts put the repo root on `sys.path` themselves and import `RayTracing...`/`geom...` directly. Scripts under `Optimization/New/` instead import `RTWave_SHM....` and target that folder's own copy of the package.
- `tests/run_case.py --code-dir <pkg dir>` runs one MUSE case with any copy of the package (used by `tests/compare_v1_v2.py`); the reduced default case (201 rays) runs in well under a second with the current code and ~15 s with v1.
- When testing changes, reduce `n_rays`, `nfft` and the time vector length first; the full benchmark (2001 rays, 10 000 samples) takes a few seconds with the current code but minutes with v1.

## Architecture

The simulation is organized as geometry + wave propagation, glued together by `Map2D`:

- **`geom/objects_2d.py`** — geometric/material primitives:
  - `medium`: a region with a wave-speed model (`ws`, an object exposing callables `S0(freq_angle_tuple)`/`A0(freq_angle_tuple)`), thickness, damping (`xi`), and the `objs`/`sensors` it contains. Computes per-ray phase velocity (`v_ray`), dispersive FFT-domain time-shifting (`fshift_dispersion`/`fshift_nd`), and transmission-loss damping (`tl`).
  - `Segment`, `Circunf`, `Polygon`, `Circle`: boundary shapes. `Segment.intersect()` is where reflection/refraction is triggered when a ray crosses it.
  - Objects are added to a medium via `medium.add_objs(...)`; anything with a `signal` attribute is treated as a sensor rather than a boundary.
- **`geom/geom_utils.py`** — low-level 2D vector math (dot/cross/norm) and segment-segment / circle-segment intersection routines used throughout, as Numba `@jit(nopython=True, cache=True)` kernels behind thin Python wrappers. The segment test is symmetric with a 1e-9 mm tolerance; because of that, `ray_refl`/`ray_refr` spawn reflected/refracted rays 1e-8 mm off the wall along their new direction so they do not re-hit it.
- **`geom/map_2d.py`** (`Map2D`) — top-level simulation driver:
  - Primary ray store is the in-memory dict `ray_cache` keyed by `Ray.__hash__()` (`save_ray`/`get_ray`); the HDF5 file is lazily opened and only used as persistence/fallback (`flush_rays_to_h5()` is never called automatically, so rays are not written to disk unless you call it).
  - `calc_t()` advances every tracked ray hash by calling `Ray.trace()` with the cyclic GC disabled; new rays spawned during tracing (reflections, refractions, mode conversions) are appended to `self.rays_h` and processed in the same pass. A `procs` (multiprocessing) code path exists but is `NotImplementedError`.
  - `add_sensor()` requires the sensor's bounding box to be fully inside one medium's bounding box (raises `KeyError` otherwise) — a sensor cut by a boundary would only see the rays of one medium.
  - `calc_signal()` / `save_signals()` collect sensor output; `plot2d()`/`plot3d()`/`plot2d_contour()` visualize rays and boundaries with `matplotlib`.
- **`RayTracing/Ray.py`**:
  - `Ray`: a single ray of one Lamb-wave `kind` (`'S0'`/`'A0'`), a `__slots__` class since huge numbers of rays are created. Tracks per-segment history (`trace_points`, `d`, `x`, `int_times`, `freq`, `a`) as parallel lists indexed by intersection event, not by time step. `trace()` advances the ray, checks intersections against all `medium.sensors` and `medium.objs` (first `objs` hit stops the loop — only one boundary interaction per step), and returns hashes of any newly spawned rays. `calc_ray()` does the actual linear propagation plus dispersive frequency shift (`medium.fshift`) and transmission-loss amplitude decay (`medium.tl`).
  - `alive_ray()` / `ray.alive`: a ray dies (stops propagating) once its amplitude drops below `a_tol` or its computed intersection time goes backwards (see `calc_ray`'s `t < t0` branch). Daughter rays whose birth amplitude is already below `a_tol` are never created (`ray_refl`/`ray_refr`).
  - Per-ray cached invariants set in `__init__`: `_hash`, `_dom_freq_idx`/`_dom_freq` (dominant FFT bin, invariant because `fshift` only rotates phases), `fft_speed` and `_phase_coeff = -2j*pi*fft_freq/fft_speed` (built by `_dispersion_for_direction()` with one batched Numba interpolation when `medium.ws` has `batch_speed`). `fft_freq` is shared by reference from the `Beam` (`_fft_freq` kwarg).
  - `dispersion_follows_direction` (module switch, default `True`): after a reflection `ray_refl` calls `_update_phase_coeff()` so `fft_speed`/`_phase_coeff` describe the **last** segment's direction. Anything that needs the coefficient of an earlier segment must use `ray.phase_coeff_at(i)`. With the switch `False` the birth-direction curve is kept for life (v1 behaviour).
  - `Beam` / `Beam_from_pzt`: generate a fan of rays (default 'circ' emission pattern) at a given center frequency/burst shape (`RayTracing/Signal.py`), producing both S0 and A0 rays per direction unless `kind` restricts to one. The burst has unit amplitude; each ray carries `a0 = power/(2*n_rays)`, so signals scale linearly with `power` (v1 scaled with `a0**2`).
  - Ray identity/dedup and HDF5 (de)serialization is by `Ray.__hash__()`, computed from kind + initial direction + initial position + birth time + initial frequency content — *not* a random or incrementing id.
- **`RayTracing/Sensors.py`** (`Sensor`) — rectangular/square/circular receiver regions built from boundary primitives (`_StrBoundary`/`_CircBoundary`, which add sensor-only intersection logic on top of `Segment`/`Circunf`). `Sensor.intersect()` just records where rays cross the sensor boundary (doesn't perturb the ray); `Sensor.signal()` later integrates the recorded ray crossings (in pairs — a ray must cross the sensor boundary twice) into a time-domain signal, weighted by an integration `window` (`'hsphere'` by default). `_signal_on_ray()` samples each chord every `d_x` (0.1 mm; coarser steps were measured to give ~4 % error), assigns every point to the ray event it lies in, drops points beyond the ray's last position, and advances phase/damping along each run of same-segment points by a multiplicative step recurrence (`_integrate_run`). `tests/test_signal_on_ray.py` pins these semantics against a per-point reference.
- **`utils_rays/ray_utils.py`** — the reflection/refraction/mode-conversion physics used by `Segment.intersect()`:
  - `ray_refl`: specular reflection, splits energy into a same-mode reflected ray (mutates the incident ray's params in place, `i=-1`) and a `mode_change` ray (opposite S0/A0 kind).
  - `ray_refr`: Snell's-law refraction into an adjacent medium (`m2`), only exists when `|sin_theta_r| <= 1`; also spawns a mode-converted refracted ray. Its docstring says it must run before reflection, but `Segment.intersect()` actually calls `ray_refl` first and passes the *pre-reflection* parameters (`ray_params_i`) explicitly; the only leftover is that the Snell velocity ratio `m2.v_ray(ray)/ray.medium.v_ray(ray)` is evaluated with the already-reflected `ray.d[-1]` (harmless for isotropic media, a small pre-existing inaccuracy for the anisotropic tables).
  - `mode_change`: spawns a new `Ray` of the opposite Lamb mode kind, carrying part of the incident energy.
  - `save_ray`/`load_ray`: HDF5 (de)serialization of a ray's full history plus attrs (`medium` hash, `kind`, `parent` hash, `alive`).
- **`utils_rays/aux_functions.py`, `utils_rays/math_utils.py`, `utils_rays/Constants_PZT.py`, `utils_rays/read_abq_CE2tests.py`** — supporting numerics (e.g. simple time-integration helpers) and material/PZT constants used by example wave-speed models; not part of the core ray-tracing engine.

### Key invariants / gotchas when modifying this code

- Ray history arrays (`trace_points`, `d`, `x`, `int_times`, `freq`, `a`) on `Ray` must stay index-aligned; `set_param(..., i=None)` appends a new event, `i=<int>` overwrites an existing one (used when finalizing a ray's parameters exactly at an intersection point after previously overshooting it).
- A ray's medium is fixed at spawn time — refraction spawns a *new* `Ray` in the new medium rather than mutating the existing one; reflection reuses the same `Ray` object in the same medium.
- `medium.objs` intersection order matters: `Ray.trace()` breaks after the first object in `medium.objs` that reports an intersection, so only one boundary crossing is handled per `trace()` call even if geometrically a ray could cross multiple boundaries in one step.
- Sensors do not affect ray propagation (`Sensor.intersect()` always returns `[]`); they are purely observational and are checked separately from `medium.objs` in `Ray.trace()`.
- `medium.__hash__()` is derived from `xi`/`th` and a random draw at construction — two `medium` instances are (intentionally) always treated as distinct, and `Map2D.mediums` / HDF5 ray attrs key off this hash, so don't rely on equal-parameter mediums comparing equal.
- `Ray._phase_coeff`/`fft_speed` are **not** constant over a ray's life when `dispersion_follows_direction` is `True` (they change at every reflection). Never cache them at ray level for signal reconstruction; use `phase_coeff_at(seg_idx)`.
- The wave-speed tables (`example/muse_stacking.hdf5`) return m/s and callers multiply by 1e3 to get mm/s (geometry is in mm, time in s). The Numba interpolators extrapolate linearly outside the tabulated `f*h` range; the damage medium (`th=3` mm) relies on this.
- Validation numbers to expect from `tests/compare_v1_v2.py` on the full benchmark: NRMSE ≈ 5e-9 with `dispersion_follows_direction=False` (residual from the 1e-8 mm boundary offset), ≈ 0.15 with it `True` (reflected arrivals change; direct-arrival windows agree to ≤ 2e-3).

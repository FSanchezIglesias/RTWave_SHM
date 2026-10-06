# Quadratic damage term scattering

- **Sesión:** `4f23fd8d-70f2-4794-b3d2-2194edc5451b`
- **Inicio de la sesión:** 2026-09-18
- **Exportado:** 2026-10-06
- **Mensajes incluidos:** 23 (4 del usuario, 19 de Claude)

> Nota: esta exportación contiene solo el texto de los mensajes. Se omiten las llamadas a herramientas (150 llamadas y sus 150 resultados), el razonamiento interno, los mensajes de subagentes, los bloques `system-reminder` y `local-command-caveat` (incluidos los comandos locales como `/model`) y las notificaciones automáticas de tareas en segundo plano. Los mensajes duplicados se han eliminado.

---

## Usuario

The current model has a very significant scattering problem. It's very, very obvious in the quadratic damage term: when the wavefront reaches the corner, a very large wave discontinuity is generated. Can you study this and look for solutions?

## Claude

I'll study how rays behave at the damage corners (reflection, mode conversion, sensor integration) and what the physics is missing there, then draft a plan. First, exploration.

## Claude

The exploration confirms the model is pure geometrical optics: no edge diffraction, angle-independent coefficients, and a 1e-9 mm corner tie. Let me read the key code paths myself before designing.

## Claude

Next I want the run template, the plate parameters, the wave-speed API, the Map2D driver and my memory notes, so I can run a small diagnostic reproduction and quantify the jump.

## Claude

The shell choked on the long heredoc; I'll write the patch script to the scratchpad with the Write tool and run it from there.

## Claude

The file has no trailing newline after `return ray`; I'll match without it and rerun only the remaining patches (Ray.py was already written).

## Claude

Simple bug: I compared against the tuple instead of its distance. Fixing and rerunning the three suites.

## Claude

Patches applied. Now running three independent checks in parallel: the corner probes with ray count, the half-plane comparison numbers, and the regenerated snapshot.

## Claude

The v1 regression is unchanged at 4.8e-9. Now the documentation: I'll read the README tail and the CLAUDE.md anchors to place an "Edge diffraction" section and the new invariants.

## Claude

Waiting for the video render to finish (about 3 minutes left); I'll monitor its log for the completion line, then extract the corresponding "after" frame.

## Claude

The background wait task will notify me when the render is done; there is nothing further to request until then.

## Claude

Applying the spectral-jump patch, then I'll rerun the diffraction tests, the transparent-damage suite, the beam-edge profiles and the corner probes.

## Claude

Everything else is done; only the final video render remains (about 6 minutes), and I will extract a frame from it once the completion notice arrives.

## Claude

Done. Summary of the study, the fix and its verification.

**Cause.** The solver was pure geometrical optics: constant reflection/transmission ratios and no edge radiation, so every corner cast a mathematically sharp shadow. Measured on the 48 mm square with 4 mm probes 270 mm from the source, the ratio to the intact plate jumped from 0.00 to 1.00 within 2 degrees across the corner line. At 350 kHz the S0 wavelength is 14.9 mm, so the Fresnel number of that damage is only 0.4 at 100 mm behind it; physics smooths such a shadow over roughly 40 mm. The circle has the same hard tangent shadow plus a lens caustic behind it.

**Fix (UTD-style corner diffraction, on by default).**
- `build_vertices` in [objects_2d.py](geom/objects_2d.py) finds the corners (shared segment endpoints, free edges); the plate outline never diffracts.
- In `Ray.trace`, exactly one ray per family (the one passing within half a ray spacing of a vertex) calls `ray_diff` in [ray_utils.py](utils_rays/ray_utils.py). Two probe rays straddling the vertex are traced through the obstacle walls carrying the ray's spectrum, and every outgoing family whose spectrum differs between the probes gets a fan of rays from the vertex with half the difference spectrum, negative where the family exists and positive in its shadow. A transparent wall yields no fans, and the delay/chirp of the wave transmitted through the 3 mm region is compensated with its real waveform.
- Fan rays carry an amplitude law `ρ/(X+ρ)·T(w)` ([utd.py](utils_rays/utd.py), Fresnel transition at the dominant frequency) applied by the sensor integration and renderers, so the fans stay continuous with the model's density-based spreading. Fans are dense near the boundary and sparse far out; the video renderer now spreads each ray over its tube width.

**Verification.**
- Rigid half-plane versus the exact Sommerfeld solution (350 kHz component): 0.59 on the shadow boundary (exact 0.46 to 0.53), 0.42 / 0.30 / 0.20 at 5 / 15 / 25 degrees into the shadow (exact 0.37 / 0.19 / 0.19), 0.73 on the reflection boundary (exact 0.68).
- MUSE square, ratio across the corner line from −6 to +6 degrees: 0.28, 0.42, 0.46, 0.47 | 0.50, 0.56, 0.59, 0.66, 0.78 (before: 0.35, 0.34, 0.08, 0.00 | 1.00 ×4). New [test_diffraction.py](tests/test_diffraction.py) covers both; all existing suites pass; the v1 and cells/holes regressions run with the new `--no-diffraction` flag and are unchanged (4.8e-9, 8.3e-9).
- Cost on the full benchmark: 13.6k rays and 2.6 s versus 4.8k rays and 1.0 s.
- The square video was re-rendered to `example/videos/D-48.0_X-355.0_Y-147.0_holes_nfft500.mp4`. At the full-plate scale the change is subtle (faint diffracted arcs off the corners, softer beam edges); the sensor probes are the clearer evidence.

**Limitations left open.** The ellipse's tangent shadow and lens caustic (stage 2, Gaussian beams), edge mode conversion, frequency dependence of the transition, and second-order diffraction. The penetrable-corner treatment is heuristic and validated on the rigid edge plus continuity; near the transmitted beam it produces destructive interference between the delayed transmitted wave and the incident-like diffracted wave, which is expected physically but not independently verified. Nothing is committed; README and CLAUDE.md document the feature.

## Usuario

What do you recommend me to improve the behaviour?

## Claude

Ranked by value for what you actually use (sensor database and videos):

1. **Validate the penetrable corner against a reference, not only the rigid edge.** The rigid half-plane test proves the machinery; the damage corner adds transmission, a half-period delay and dispersion, which I only checked for continuity. A single 2D FEM run (Abaqus, one source, the 48 mm square) with a line of receivers behind the damage would tell you whether the current interference between the delayed transmitted wave and the diffracted wave is right or needs a different weighting. This is the one thing I could not verify.

2. **Use the physical reflection and transmission coefficients at the faces instead of constants.** `ratio_rfl` and `boundary_losses` are angle- and frequency-independent, so the jumps that drive the fans are only as good as those numbers. Lamb-wave scattering at a thickness step (S0 and A0 at a 1.3 to 3 mm step) has published coefficients; even a simple obliquity model that makes transmission vanish at grazing incidence would improve the grazing zone between the two corner lines, where the model now relies on total internal reflection.

3. **Frequency-dependent transition.** `T(w)` is evaluated at 350 kHz for all bins, so diffracted pulses keep the incident shape. Evaluating `w` per bin costs one extra vector per point and matters for the A0 mode, whose wavelength is three times shorter and whose transition zone is therefore narrower.

4. **Gaussian-beam receiver model (stage 2).** It is the only route that also fixes the ellipse's tangent shadow and the caustic behind it, and it regularises the speckle you see in the far field of the videos. Do it as a switchable sensor model and re-validate the intact plate against the chord model before adopting it.

5. **Mode conversion at edges.** Corners convert S0 to A0 and back; the fans keep the incident mode. Straightforward to add once you have a coefficient (item 2) to size it.

6. **Second-order diffraction and cost.** `max_order=1` is fine for the isolated damage. If you later model stiffeners or multiple defects, allow order 2 and raise `min_rel_amp` to keep the ray count in check; the current 2.6× runtime comes mostly from edge-reflected families re-exciting the corners.

If you only do one thing, do item 1: it decides whether items 2 and 3 are refinements or corrections.

## Usuario

Apply the 1.

## Claude

Item 1 needs an independent reference solution for the penetrable corner. Let me first check whether Abaqus is available on this machine; if not, I'll build the reference myself with a 2D finite-difference wave solver on the same geometry.

## Claude

All suites pass (7 diffraction tests now including total internal reflection). Last checks: run the committed FDTD tool once from its repository location, and wait for the video render.

## Claude

The reference tool works from the repository and reproduces the numbers. Only the video render is still running; I'll report once it finishes.

## Claude

Recommendation 1 is applied, and it paid off beyond validation.

**Reference built.** No Abaqus on this machine, so I wrote an independent reference: a 2D finite-difference solution of the scalar wave equation for the same idealised problem (isotropic non-dispersive plate at 5.205 mm/µs, the 48 mm square at 4.432 mm/µs, absorbing edges, point source, 3-cycle burst). The ray tracer was run on exactly that problem with the reflection ratio the scalar impedance contrast implies (0.08). It lives in [compare_fdtd_corner.py](tests/compare_fdtd_corner.py) and runs in about two minutes.

**What it showed.**
- Across the corner shadow line the ray model with diffraction tracks the reference: RMS deviation 0.13 on the arc, versus 0.37 for pure geometrical optics. The shadow-edge regions behind the damage match as well.
- The centre of the transmitted beam did not match at first: the reference shows a 2× on-axis peak, the ray model a flat 0.9. This was not the diffraction model but a lost-energy bug in the solver: when Snell's law has no solution the transmitted share was simply dropped, so the slower square could not act as the light pipe the reference shows. Total internal reflection is now handled (`RayTracing.Ray.total_internal_reflection`, default on, also in the probe tracer), and the ray model reproduces the focusing (1.6 to 1.75 versus 2.0). The remaining difference is the sharper ray caustic, which is the stage-2 Gaussian-beam topic.

**Verification.**
- New test for the reflection rule; all suites pass (7 diffraction tests, hole mesh, ellipse, sensor integration).
- v1 comparison unchanged at 4.8e-9 using the new `--v1-physics` flag (no diffraction, energy lost at total internal reflection); cells-versus-holes unchanged at 8.3e-9.
- Full benchmark with the final physics: 17.3k rays and 3.4 s, versus 4.8k rays and 1.0 s with v1 physics.
- The square-damage video was re-rendered with the final code.

**Note for your database.** The total-internal-reflection fix changes results wherever rays hit the inner faces of the thick damage obliquely, independently of diffraction. Signals produced before today use the v1 physics; the flag lets you reproduce them.

Also worth knowing: two media created with `xi = 0` hash to the same key and the second silently replaces the first in `Map2D`. I documented it in CLAUDE.md rather than changing the hash.

## Usuario

Exporta esta conversación a Markdown y guárdala en la carpeta `conversations/` de la raíz del repo.

1. Usa la herramienta de exportación de transcripciones de la sesión (session "self"). Genera un zip en Descargas con un transcript.jsonl.
2. Extrae el zip en el directorio scratchpad, no en el repo.
3. Convierte transcript.jsonl a Markdown con un script en Python. Incluye solo el texto de los mensajes del usuario y de Claude, con secciones "## Usuario" y "## Claude". Excluye los mensajes de subagentes (isSidechain), los bloques system-reminder y local-command-caveat, el razonamiento interno y las llamadas a herramientas con su salida. Elimina mensajes duplicados.
4. Cabecera del archivo: título, id de sesión, fecha de inicio, fecha de exportación y una nota indicando que se omiten las llamadas a herramientas (con su número).
5. Nombre del archivo: `AAAA-MM-DD_titulo-corto-en-kebab-case.md`, con la fecha de inicio de la sesión. Si la carpeta no existe, créala.
6. No hagas commit. Dime la ruta del archivo, el número de mensajes y la ruta del zip original.

Si añado "con detalle", incluye también los comandos y sus resultados, truncando salidas de más de 40 líneas.

## Claude

Voy a exportar la transcripción de esta sesión; primero cargo la herramienta de exportación.

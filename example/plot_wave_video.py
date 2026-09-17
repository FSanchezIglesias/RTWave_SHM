"""
Wave propagation video — both S0 and A0 modes.

Runs the full simulation for PZT1 (source) and exports an MP4 animation
of the 2D wavefield using a precomputed frame stack for efficiency.

Strategy
--------
Rather than calling plot2d_contour() N_FRAMES times (which would recompute
irfft for every ray × trace point × frame), we make a single pass over all
rays, reconstruct the time signal at each trace point ONCE, then sample the
N_FRAMES time indices in one vectorised operation.  This reduces irfft calls
by a factor of N_FRAMES (~100-200x speedup for typical settings).
"""

import gc
import os
import sys

# Allow running from anywhere: repo root (parent of example/) for the
# `geom`/`RayTracing` packages, this folder for `MUSE_dmg`.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')           # non-interactive backend — required for video
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.fft import irfft
from tqdm import tqdm

from MUSE_dmg import gen_MUSE_dmg, gen_MUSE_intact
# NOTE: import the same package modules MUSE_dmg uses (repo-root `geom`,
# `RayTracing`), not `RTWave_SHM.geom...`: those would be distinct classes
# and the isinstance() checks in draw_overlay would silently draw nothing.
from RayTracing.Ray import Beam_from_pzt
from geom.objects_2d import Segment, Ellipse

# ---------------------------------------------------------------------------
# Matplotlib style — LaTeX text rendering (match plot_intact_signals.py)
# ---------------------------------------------------------------------------
USE_LATEX = True
plt.rcParams.update({
    "text.usetex": USE_LATEX,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# ---------------------------------------------------------------------------
# Simulation parameters  (match 02_Run-Single-Damage.py)
# ---------------------------------------------------------------------------
UNDAMAGED        = False    # True → intact plate (gen_MUSE_intact), no damage zone

XDMG, YDMG       = 355., 147.
XLDMG, YLDMG     = 48.,  24.
DMG_SHAPE         = 'ellipse'         # 'rect' or 'ellipse' (see MUSE_dmg.gen_MUSE_dmg)
PHIDMG            = np.deg2rad(30.)   # ellipse rotation [rad]
THDMG             = 3.
RDMG              = 0.1
BLDMG             = 0.05
RMDMG             = 1

NRAYS             = 3_001
F                 = 350.e3                          # Hz
T                 = np.linspace(0., 0.0002, 10_000)   # s  (50 MHz)
SOURCE_IDX        = 0                               # PZT1 as source

# ---------------------------------------------------------------------------
# Video / grid parameters
# ---------------------------------------------------------------------------
N_FRAMES    = 200        # number of animation frames (subsampled from 10 000)
GRIDLEN     = 2.         # mm per grid cell — coarser = faster precomputation
FPS         = 5          # output frames per second
_VIDEO_DIR  = os.path.join(_THIS_DIR, 'videos')
if UNDAMAGED:
    OUTPUT = os.path.join(_VIDEO_DIR, 'Intact.mp4')
elif DMG_SHAPE == 'ellipse':
    OUTPUT = os.path.join(
        _VIDEO_DIR,
        f'E-{XLDMG}x{YLDMG}_A-{np.degrees(PHIDMG):.0f}_X-{XDMG}_Y-{YDMG}.mp4')
else:
    OUTPUT = os.path.join(_VIDEO_DIR, f'D-{XLDMG}_X-{XDMG}_Y-{YDMG}.mp4')
CMAP        = 'Spectral_r'  # diverging colourmap

# Set to True to save the precomputed frame stack to disk as a .npz archive.
# The archive contains:
#   z_frames  — float32 (N_FRAMES, ngridx, ngridy) amplitude array
#   t_frames  — float64 (N_FRAMES,) time values [s] for each frame
# Reload with: data = np.load(FRAMES_NPZ); z = data['z_frames']
SAVE_FRAMES = False
FRAMES_NPZ  = os.path.join(_VIDEO_DIR, 'z_frames.npz')

# ---------------------------------------------------------------------------
# Plate bounds
# ---------------------------------------------------------------------------
L = 726.
XMIN, XMAX = 0., L
YMIN, YMAX = 0., L


def _grid_index(tr: np.ndarray,
                xmin: float, xmax: float, ngridx: int,
                ymin: float, ymax: float, ngridy: int) -> tuple[int, int]:
    """Map a 2-D trace point to grid indices (zi, zk)."""
    zi = int((tr[0] - xmin) / (xmax - xmin) * ngridx)
    zk = int((tr[1] - ymin) / (ymax - ymin) * ngridy)
    return zi, zk


def precompute_frames(
        ray_map, frame_t_indices: np.ndarray,
        ngridx: int, ngridy: int
    ) -> np.ndarray:
    """
    Build a (N_FRAMES, ngridx, ngridy) amplitude array in one ray pass.

    Samples uniformly along each ray SEGMENT (not just at the discrete
    trace_point endpoints), so the entire propagation path is captured.
    Uses step-recurrence for phase and damping (same strategy as
    Sensor._signal_on_ray) to avoid recomputing exp() per sample.
    """
    n_frames = len(frame_t_indices)
    n_t      = len(ray_map.init_beam.t)
    t_vec    = ray_map.init_beam.t
    d_x      = GRIDLEN          # sample spacing along ray = one grid cell [mm]

    z_frames = np.zeros((n_frames, ngridx, ngridy), dtype=np.float32)

    for rh in tqdm(ray_map.rays_h, desc='Precomputing frames', unit='ray'):
        ray = ray_map.get_ray(rh)

        for seg_idx in range(len(ray.trace_points) - 1):
            x_start = ray.x[seg_idx]
            x_end   = ray.x[seg_idx + 1]
            if x_end <= x_start:
                continue

            # Segment-level constants (computed once per segment)
            f0           = ray.freq[seg_idx]
            a0           = ray.a[seg_idx]
            t0           = ray.int_times[seg_idx]
            x0           = ray.x[seg_idx]
            v            = ray.medium.v_ray(ray, seg_idx)

            # Sample positions on a GLOBAL time lattice (spacing d_x / v) rather
            # than every d_x from the segment start: rays split at a wall (e.g.
            # a transparent internal wall) then sample exactly the same points
            # as the unsplit ray, so wavefields of different meshes can be
            # compared cell by cell without binning speckle.
            dt_s      = d_x / v
            k0        = int(np.ceil((t0 + 1e-15) / dt_s))
            t_samples = np.arange(k0, k0 + int((x_end - x_start) / d_x) + 2) * dt_s
            x_samples = x0 + (t_samples - t0) * v
            x_samples = x_samples[(x_samples > x_start) & (x_samples < x_end)]
            if x_samples.size == 0:
                continue
            damping_rate = 2.0 * np.pi * ray._dom_freq * ray.medium.xi / v

            # Step-recurrence initialisaton — advance phase/amp by d_x each step.
            # Dispersion curve of THIS segment's direction (ray._phase_coeff
            # only describes the last segment when dispersion follows direction).
            phase_coeff = ray.phase_coeff_at(seg_idx)
            dx_first   = x_samples[0] - x0
            step_phase = np.exp(phase_coeff * d_x)               # [nfft] complex
            step_amp   = float(np.exp(-damping_rate * d_x))      # scalar
            cur_f      = np.exp(phase_coeff * dx_first) * f0
            cur_amp    = float(a0 * np.exp(-damping_rate * dx_first))

            for x_s in x_samples:
                dx_k   = x_s - x0
                pos    = ray.trace_points[seg_idx] + ray.d[seg_idx] * dx_k
                zi, zk = _grid_index(pos, XMIN, XMAX, ngridx, YMIN, YMAX, ngridy)

                if 0 <= zi < ngridx and 0 <= zk < ngridy:
                    t_s  = t0 + dx_k / v
                    s    = cur_amp * irfft(cur_f, n=n_t)
                    # causality mask (t/2 accounts for dispersion smearing,
                    # matching the convention in Ray.signal_at_x)
                    s[:np.searchsorted(t_vec, t_s / 2)] = 0.0
                    z_frames[:, zi, zk] += s[frame_t_indices].astype(np.float32)

                # Advance recurrence — cheap multiply, no exp
                cur_f   *= step_phase
                cur_amp *= step_amp

    return z_frames


def draw_overlay(ax: plt.Axes, ray_map) -> None:
    """Draw damage walls, invisible mesh walls, and PZT sensor circles.

    * Blue segments / ellipse → damage boundary, rendered in red.
    * White segments → invisible internal mesh walls (including the
                       transparent cell around an elliptical damage),
                       rendered as thin dashed black lines so the mesh
                       structure is visible without dominating the wavefield.
    * Plate boundary (black) segments are skipped.

    Shared segments appear in multiple mediums' obj lists, so a ``seen``
    set deduplicates by object identity before drawing.
    """
    seen: set = set()

    for med in ray_map.mediums.values():
        for obj in med.objs:
            if not isinstance(obj, (Segment, Ellipse)) or id(obj) in seen:
                continue
            seen.add(id(obj))
            if isinstance(obj, Ellipse):
                obj.plot(ax, color='red')
                for patch in ax.patches[-1:]:
                    patch.set_linewidth(2.)
                    patch.set_zorder(4)
            elif obj.color == 'blue':
                obj.plot(ax, color='red')
            elif obj.color == 'white':
                ax.plot(
                    [obj.a1[0], obj.a2[0]],
                    [obj.a1[1], obj.a2[1]],
                    color='black', alpha=1.,
                    linewidth=2., linestyle='--',
                )
        for sens in med.sensors:
            center = sens.origin()
            r = sens.bounds[0].r
            ax.add_patch(
                plt.Circle(center, r * 1.3, color='black', fill=True, zorder=3)
            )


def save_frames(z_frames: np.ndarray, t: np.ndarray,
                frame_t_indices: np.ndarray) -> None:
    """Save the precomputed frame stack to a compressed NumPy archive.

    Parameters
    ----------
    z_frames:
        Float32 array of shape (N_FRAMES, ngridx, ngridy).
    t:
        Full simulation time vector [s].
    frame_t_indices:
        Integer indices into *t* that select each frame.
    """
    os.makedirs(os.path.dirname(FRAMES_NPZ) or '.', exist_ok=True)
    np.savez_compressed(
        FRAMES_NPZ,
        z_frames=z_frames,
        t_frames=t[frame_t_indices],
    )
    print(f'Frame stack saved -> {FRAMES_NPZ}')
    print(f'  shape : {z_frames.shape}  (N_FRAMES x ngridx x ngridy)')
    print(f'  dtype : {z_frames.dtype}')
    print(f'  reload: np.load("{FRAMES_NPZ}")["z_frames"]')


def make_video(z_frames: np.ndarray, t: np.ndarray,
               frame_t_indices: np.ndarray, ray_map,
               ngridx: int, ngridy: int) -> None:
    """Render and save the animation as MP4."""

    # Colour scale: 98th percentile avoids outliers dominating the range
    lvl_lim = float(np.nanpercentile(np.abs(z_frames), 98))
    if lvl_lim == 0.:
        lvl_lim = 1.

    fig, ax = plt.subplots(figsize=(9, 8), facecolor='white')
    ax.set_facecolor('white')
    # Reserve extra right-hand margin so the (larger) colorbar label isn't
    # clipped by the figure edge — subplots() default margins were sized
    # for the smaller fontsize this figure used to use.
    fig.subplots_adjust(left=0.09, right=0.86, bottom=0.08, top=0.95)

    # Initial imshow — shape must be (ngridy, ngridx) for correct x/y axes
    im = ax.imshow(
        z_frames[0].T,
        origin='lower',
        extent=[XMIN, XMAX, YMIN, YMAX],
        vmin=-lvl_lim, vmax=lvl_lim,
        cmap=CMAP,
        interpolation='bilinear',
        aspect='equal',
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'\textit{Amplitude} [a.u.]', color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')

    # Static overlay: damage square + sensor circles (drawn once, on top)
    draw_overlay(ax, ray_map)

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel(r'\textit{X} [mm]', color='black')
    ax.set_ylabel(r'\textit{Y} [mm]', color='black')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    def update(fi: int):
        im.set_data(z_frames[fi].T)
        return [im]

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(frame_t_indices),
        interval=1000 / FPS,
        blit=True,
    )

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    try:
        writer = animation.FFMpegWriter(
            fps=FPS, bitrate=3000,
            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'],
        )
        ani.save(OUTPUT, writer=writer, dpi=120)
        print(f'MP4 saved -> {OUTPUT}')
    except Exception as e:
        print(f'ffmpeg not available ({e}). Falling back to GIF...')
        gif_path = OUTPUT.replace('.mp4', '.gif')
        ani.save(gif_path, writer='pillow', fps=FPS, dpi=80)
        print(f'GIF saved  -> {gif_path}')

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
NFFT = 500
MESH = 'holes'      # gen_MUSE_dmg mesh: 'holes' (default) or legacy 'cells'


def _video_name() -> str:
    if UNDAMAGED:
        return os.path.join(_VIDEO_DIR, 'Intact.mp4')
    if DMG_SHAPE == 'ellipse':
        base = f'E-{XLDMG}x{YLDMG}_A-{np.degrees(PHIDMG):.0f}_X-{XDMG}_Y-{YDMG}'
    else:
        base = f'D-{XLDMG}_X-{XDMG}_Y-{YDMG}'
    return os.path.join(_VIDEO_DIR, f'{base}_{MESH}_nfft{NFFT}.mp4')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='MUSE wavefield video')
    ap.add_argument('--shape', choices=['rect', 'ellipse'], default=DMG_SHAPE)
    ap.add_argument('--nfft', type=int, default=NFFT)
    ap.add_argument('--mesh', choices=['holes', 'cells'], default=MESH)
    ap.add_argument('--intact', action='store_true')
    ap.add_argument('--out', default=None, help='output file (default: derived from the parameters)')
    _a = ap.parse_args()
    DMG_SHAPE, NFFT, MESH = _a.shape, _a.nfft, _a.mesh
    UNDAMAGED = UNDAMAGED or _a.intact
    OUTPUT = _a.out or _video_name()

    # 1 — Build geometry
    print('Building geometry...')
    if UNDAMAGED:
        m, pzts = gen_MUSE_intact()
    else:
        m, pzts = gen_MUSE_dmg(
            xdmg=XDMG, ydmg=YDMG, xldmg=XLDMG, yldmg=YLDMG,
            thdmg=THDMG, rdmg=RDMG, bldmg=BLDMG, rmdmg=RMDMG,
            shape=DMG_SHAPE, phidmg=PHIDMG, mesh=MESH,
        )

    # 2 — Initial beam  (both S0 and A0)
    print(f'Emitting beam from PZT{SOURCE_IDX + 1} ({NRAYS} rays, S0+A0)...')
    ibeam = Beam_from_pzt(
        NRAYS, pzts[SOURCE_IDX], power=2001 / 8,
        f=F, npeaks=3, nfft=NFFT, t=T,
    )
    m.set_init_beam(ibeam)

    # 3 — Ray tracing
    print('Tracing rays...')
    m.calc_t()
    gc.collect()

    # 4 — Grid setup
    ngridx = int((XMAX - XMIN) / GRIDLEN)
    ngridy = int((YMAX - YMIN) / GRIDLEN)
    print(f'Grid: {ngridx} × {ngridy}  ({GRIDLEN} mm/cell)')

    # 5 — Frame time indices (subsample time axis)
    frame_t_indices = np.linspace(0, len(T) - 1, N_FRAMES, dtype=int)

    # 6 — Precompute all frames (single ray pass)
    z_frames = precompute_frames(m, frame_t_indices, ngridx, ngridy)
    gc.collect()

    # 7 — Optionally persist the frame stack
    if SAVE_FRAMES:
        save_frames(z_frames, T, frame_t_indices)

    # 8 — Render and export
    print('Rendering video...')
    make_video(z_frames, T, frame_t_indices, m, ngridx, ngridy)

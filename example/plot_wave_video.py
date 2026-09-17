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

import matplotlib
matplotlib.use('Agg')           # non-interactive backend — required for video
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.fft import irfft
from tqdm import tqdm

# Allow running from anywhere: make sure the repo root (parent of this
# example/ folder) is importable as the top-level `geom`/`RayTracing`
# packages, and that this folder itself is importable for `MUSE_dmg`.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from MUSE_dmg import gen_MUSE_dmg
from RayTracing.Ray import Beam_from_pzt
from geom.objects_2d import Segment
from plate_config import (
    L, XDMG, YDMG, XLDMG, YLDMG, THDMG, RDMG, BLDMG, RMDMG,
    NRAYS, F, T, SOURCE_IDX,
)

# ---------------------------------------------------------------------------
# Simulation parameters — sourced from plate_config.py, shared with
# Run-Single-Damage.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Video / grid parameters
# ---------------------------------------------------------------------------
N_FRAMES    = 200        # number of animation frames (subsampled from 10 000)
GRIDLEN     = 2.         # mm per grid cell — coarser = faster precomputation
FPS         = 5          # output frames per second
VIDEOS_DIR  = os.path.join(_THIS_DIR, 'videos')
OUTPUT      = os.path.join(VIDEOS_DIR, f'D-{XLDMG}_X-{XDMG}_Y-{YDMG}.mp4')
CMAP        = 'Spectral_r'  # diverging colourmap

# Set to True to save the precomputed frame stack to disk as a .npz archive.
# The archive contains:
#   z_frames  — float32 (N_FRAMES, ngridx, ngridy) amplitude array
#   t_frames  — float64 (N_FRAMES,) time values [s] for each frame
# Reload with: data = np.load(FRAMES_NPZ); z = data['z_frames']
SAVE_FRAMES = False
FRAMES_NPZ  = os.path.join(VIDEOS_DIR, 'z_frames.npz')

# ---------------------------------------------------------------------------
# Plate bounds
# ---------------------------------------------------------------------------
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

            # Sample positions centred in each grid cell along the segment
            x_samples = np.arange(x_start + d_x / 2, x_end, d_x)
            if x_samples.size == 0:
                continue

            # Segment-level constants (computed once per segment).
            # RTWave_SHM's Ray only stores f0/a0/t0 at each segment start —
            # it has no precomputed "dominant frequency" or "phase
            # coefficient" fields, so we derive the per-mm step multipliers
            # here ourselves, matching medium.fshift_dispersion() and
            # medium.tl() exactly (geom/objects_2d.py):
            #   fshift_dispersion: f_d = exp(-1j*2*pi*fft_freq*x/fft_speed)*f0
            #   tl:                a_d = a0 * exp(-2*pi*f_dom*xi*(t-t0))
            #                      with f_dom = fft_freq[argmax(|freq[i]|)]
            # This assumes a dispersive medium (medium(..., dispersive=True),
            # the default and what gen_MUSE_dmg always builds); for a
            # non-dispersive medium these step multipliers would not match
            # medium.fshift_nd().
            f0           = ray.freq[seg_idx]
            a0           = ray.a[seg_idx]
            t0           = ray.int_times[seg_idx]
            x0           = ray.x[seg_idx]
            v            = ray.medium.v_ray(ray, seg_idx)
            dom_freq     = ray.fft_freq[np.argmax(np.abs(f0))]
            phase_coeff  = -1j * 2 * np.pi * ray.fft_freq / ray.fft_speed  # [nfft], per mm
            damping_rate = 2.0 * np.pi * dom_freq * ray.medium.xi / v      # scalar, per mm

            # Step-recurrence initialisaton — advance phase/amp by d_x each step
            dx_first   = x_samples[0] - x0
            step_phase = np.exp(phase_coeff * d_x)                # [nfft] complex
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

    * Blue segments  → damage walls, rendered in dodgerblue.
    * White segments → invisible internal mesh walls, rendered as thin
                       dashed white lines (alpha=0.35) so the mesh
                       structure is visible without dominating the wavefield.
    * Plate boundary (black) segments are skipped.

    Shared segments appear in multiple mediums' obj lists, so a ``seen``
    set deduplicates by object identity before drawing.
    """
    seen: set = set()

    for med in ray_map.mediums.values():
        for obj in med.objs:
            if not isinstance(obj, Segment) or id(obj) in seen:
                continue
            seen.add(id(obj))
            if obj.color == 'blue':
                obj.plot(ax, color='dodgerblue')
            elif obj.color == 'white':
                ax.plot(
                    [obj.a1[0], obj.a2[0]],
                    [obj.a1[1], obj.a2[1]],
                    color='black', alpha=1.,
                    linewidth=2., linestyle='-',
                )
        for sens in med.sensors:
            sens.plot(ax, color='white', marker='o')


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
    print(f'  shape : {z_frames.shape}  (N_FRAMES × ngridx × ngridy)')
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

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#111111')
    ax.set_facecolor('#111111')

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
    cbar.set_label('Amplitude [a.u.]', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # Static overlay: damage square + sensor circles (drawn once, on top)
    draw_overlay(ax, ray_map)

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel('x [mm]', color='white')
    ax.set_ylabel('y [mm]', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    title = ax.set_title('', color='white', fontsize=11, pad=8)

    def update(fi: int):
        im.set_data(z_frames[fi].T)
        t_us = t[frame_t_indices[fi]] * 1.e6
        title.set_text(
            f'GLW propagation  —  S0 + A0  —  t = {t_us:.1f} µs\n'
            f'AS4/8552 (+45,−45,90,0)  |  f = {F/1e3:.0f} kHz'
        )
        return [im, title]

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
    except Exception as e:
        print(f'ffmpeg not available ({e}). Falling back to GIF...')
        gif_path = OUTPUT.replace('.mp4', '.gif')
        ani.save(gif_path, writer='pillow', fps=FPS, dpi=80)
        print(f'GIF saved  -> {gif_path}')
    else:
        print(f'MP4 saved -> {OUTPUT}')

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    # 1 — Build geometry
    print('Building geometry...')
    m, pzts = gen_MUSE_dmg(
        xdmg=XDMG, ydmg=YDMG, xldmg=XLDMG, yldmg=YLDMG,
        thdmg=THDMG, rdmg=RDMG, bldmg=BLDMG, rmdmg=RMDMG,
    )
    # m, pzts = gen_MUSE_dmg(
    #     xdmg=XDMG, ydmg=YDMG, xldmg=XLDMG, yldmg=YLDMG,
    #     thdmg=1.288,   # intact plate thickness (no local thinning)
    #     rdmg=0.0,      # damage walls fully transparent (no reflection)
    #     bldmg=0.0,     # no boundary losses at damage walls
    #     rmdmg=1.0,     # no mode conversion at damage walls
    #     xi_dmg=1.e-3,  # same damping as surrounding plate
    # )

    # 2 — Initial beam  (both S0 and A0)
    print(f'Emitting beam from PZT{SOURCE_IDX + 1} ({NRAYS} rays, S0+A0)...')
    ibeam = Beam_from_pzt(
        NRAYS, pzts[SOURCE_IDX], power=NRAYS / len(pzts),
        f=F, npeaks=3, nfft=500, t=T,
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

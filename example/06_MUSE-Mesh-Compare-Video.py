"""
Side-by-side wavefield video: legacy convex-cell mesh (invisible walls) vs
the 2-medium hole mesh of ``gen_MUSE_dmg``.

Runs the same simulation twice (``mesh='cells'`` and ``mesh='holes'``) and
renders, for every frame, three wavefield panels — cells, holes and their
difference (amplified) — plus the signal of one sensor behind the damage for
both meshes with a moving time cursor.  Reuses the frame precomputation and
overlay drawing of ``plot_wave_video.py``.

Run from anywhere::

    python example/06_MUSE-Mesh-Compare-Video.py
"""

import gc
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# plot_wave_video sets the matplotlib style at import time
import plot_wave_video as pwv
from plot_wave_video import precompute_frames, draw_overlay, XMIN, XMAX, YMIN, YMAX
from MUSE_dmg import gen_MUSE_dmg
from RayTracing.Ray import Beam_from_pzt
import RayTracing.Ray as _ray_mod

# mesh-equivalence video: the cell mesh has no diffracting vertices (its
# corners are shared by 3+ walls), so compare both meshes without diffraction
_ray_mod.diffraction = False

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
XDMG, YDMG   = 363., 149.            # damage on the PZT1 -> PZT5 line
XLDMG, YLDMG = 24., 24.
DMG_SHAPE    = 'rect'                # 'rect' or 'ellipse'
PHIDMG       = np.deg2rad(30.)
THDMG        = 3.
RDMG         = 0.1
BLDMG        = 0.05
RMDMG        = 1

NRAYS        = 2_001
F            = 350.e3
T            = np.linspace(0., 0.0002, 10_000)
SOURCE_IDX   = 0                     # PZT1
SIGNAL_PZT   = 4                     # PZT5: directly behind the damage

# ---------------------------------------------------------------------------
# Video parameters
# ---------------------------------------------------------------------------
N_FRAMES   = 150
FPS        = 5
DIFF_GAIN  = None                    # amplification of the difference panel (None = auto, power of 10)
CMAP       = 'Spectral_r'
OUTPUT     = os.path.join(_THIS_DIR, 'videos',
                          f'Mesh-Compare_{DMG_SHAPE}_D-{XLDMG:.0f}x{YLDMG:.0f}_X-{XDMG:.0f}_Y-{YDMG:.0f}.mp4')


def run(mesh: str):
    """Trace the beam on one mesh; return (map, sensors)."""
    m, pzts = gen_MUSE_dmg(xdmg=XDMG, ydmg=YDMG, xldmg=XLDMG, yldmg=YLDMG,
                           thdmg=THDMG, rdmg=RDMG, bldmg=BLDMG, rmdmg=RMDMG,
                           shape=DMG_SHAPE, phidmg=PHIDMG, mesh=mesh)
    beam = Beam_from_pzt(NRAYS, pzts[SOURCE_IDX], power=2001 / 8,
                         f=F, npeaks=3, nfft=500, t=T)
    m.set_init_beam(beam)
    print(f'[{mesh}] {len(m.mediums)} mediums — tracing {NRAYS} rays...')
    m.calc_t()
    m.calc_signal()
    print(f'[{mesh}] {len(m.rays_h)} rays traced')
    return m, pzts


def make_video(frames: dict, sig: dict, maps: dict, frame_idx: np.ndarray) -> None:
    lvl = float(np.nanpercentile(np.abs(frames['cells']), 98)) or 1.
    raw = frames['holes'] - frames['cells']
    gain = DIFF_GAIN
    if gain is None:
        mx = float(np.abs(raw).max())
        gain = 10. ** np.floor(np.log10(lvl / mx)) if mx > 0 else 1.
    diff = raw * gain
    lvl_d = float(np.nanpercentile(np.abs(diff), 99)) or 1.
    t_us = T * 1e6

    fig = plt.figure(figsize=(18, 10.5), facecolor='white')
    gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.], left=0.04, right=0.955,
                          bottom=0.07, top=0.94, wspace=0.12, hspace=0.25)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_sig = fig.add_subplot(gs[1, :])

    ims = []
    panels = [('cells', frames['cells'], lvl, 'Cell mesh (invisible walls)'),
              ('holes', frames['holes'], lvl, 'Hole mesh (2 mediums)'),
              ('diff', diff, lvl_d, rf'Difference $\times 10^{{{int(round(np.log10(gain)))}}}$')]
    for ax, (key, z, vlim, title) in zip(axes, panels):
        im = ax.imshow(z[0].T, origin='lower', extent=[XMIN, XMAX, YMIN, YMAX],
                       vmin=-vlim, vmax=vlim, cmap=CMAP, interpolation='bilinear',
                       aspect='equal')
        ims.append(im)
        draw_overlay(ax, maps['cells' if key == 'cells' else 'holes'])
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(YMIN, YMAX)
        ax.set_title(title)
        ax.set_xlabel(r'\textit{X} [mm]')
        if key == 'cells':
            ax.set_ylabel(r'\textit{Y} [mm]')
        else:
            ax.set_yticklabels([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, format='%.1e' if key == 'diff' else None)
        cb.ax.tick_params(labelsize=11)

    name = f'PZT{SIGNAL_PZT + 1}'
    ax_sig.plot(t_us, sig['cells'], 'k-', lw=1.2, label='cells')
    ax_sig.plot(t_us, sig['holes'], 'r--', lw=1.0, label='holes')
    ax_sig.set_xlim(t_us[0], t_us[-1])
    ax_sig.set_xlabel(r'\textit{t} [$\mu$s]')
    ax_sig.set_ylabel(rf'{name} [a.u.]')
    ax_sig.legend(loc='upper right', ncol=2)
    nrmse = np.sqrt(np.mean((sig['holes'] - sig['cells']) ** 2)) / np.sqrt(np.mean(sig['cells'] ** 2))
    ax_sig.text(0.01, 0.95, f'NRMSE holes vs cells: {nrmse:.1e}', transform=ax_sig.transAxes,
                va='top', ha='left')
    cursor = ax_sig.axvline(t_us[0], color='b', lw=1.5)
    title = fig.suptitle('')

    def update(fi: int):
        for im, (_, z, _, _) in zip(ims, panels):
            im.set_data(z[fi].T)
        tk = t_us[frame_idx[fi]]
        cursor.set_xdata([tk, tk])
        title.set_text(rf'$t = {tk:.1f}\,\mu$s')
        return ims + [cursor, title]

    ani = animation.FuncAnimation(fig, update, frames=len(frame_idx),
                                  interval=1000 / FPS, blit=False)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    try:
        writer = animation.FFMpegWriter(fps=FPS, bitrate=4000,
                                        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        ani.save(OUTPUT, writer=writer, dpi=100)
        print(f'MP4 saved -> {OUTPUT}')
    except Exception as e:
        print(f'ffmpeg not available ({e}). Falling back to GIF...')
        gif = OUTPUT.replace('.mp4', '.gif')
        ani.save(gif, writer='pillow', fps=FPS, dpi=70)
        print(f'GIF saved  -> {gif}')
    plt.close(fig)


if __name__ == '__main__':
    ngridx = int((XMAX - XMIN) / pwv.GRIDLEN)
    ngridy = int((YMAX - YMIN) / pwv.GRIDLEN)
    frame_idx = np.linspace(0, len(T) - 1, N_FRAMES, dtype=int)

    frames, sig, maps = {}, {}, {}
    for mesh in ('cells', 'holes'):
        m, pzts = run(mesh)
        frames[mesh] = precompute_frames(m, frame_idx, ngridx, ngridy)
        sig[mesh] = pzts[SIGNAL_PZT].signal_s
        maps[mesh] = m
        gc.collect()

    d = frames['holes'] - frames['cells']
    print('max |holes - cells| on the grid: {:.3e}  (cells peak {:.3e})'.format(
        np.abs(d).max(), np.abs(frames['cells']).max()))
    print('Rendering video...')
    make_video(frames, sig, maps, frame_idx)

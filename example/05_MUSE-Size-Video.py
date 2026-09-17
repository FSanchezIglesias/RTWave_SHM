"""
MUSE damage-size growth video.

Grows a square damage zone centered on the plate (X = Y = L/2) from 8 mm to
28 mm in 0.5 mm steps, and renders the resulting gen_MUSE_dmg() mesh at
each size on a single full-plate (726x726 mm) view.

Same mesh-drawing convention as 04_MUSE-Positions-Video.py: damage box in
red, invisible internal mesh walls as dashed black lines.
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend — required for video
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

from MUSE_dmg import gen_MUSE_dmg, pzt_pos, r_pzt, l as PLATE_L
from geom.objects_2d import Segment, Ellipse

plt.rcParams.update({
    "text.usetex": True,                # Use LaTeX for all text
    "font.family": "serif",             # Serif font for math-like appearance
    "font.serif": ["Times", "Computer Modern Roman"],
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# ---------------------------------------------------------------------------
# Damage-size sweep, centered on the plate
# ---------------------------------------------------------------------------
XDMG, YDMG  = PLATE_L / 2., PLATE_L / 2.
SIZE_MIN    = 8.
SIZE_MAX    = 28.
SIZE_STEP   = 0.5
SIZES       = np.arange(SIZE_MIN, SIZE_MAX + SIZE_STEP / 2, SIZE_STEP)

# ---------------------------------------------------------------------------
# Video parameters
# ---------------------------------------------------------------------------
FRAME_SECONDS = 0.1
FPS           = 1. / FRAME_SECONDS
OUTPUT        = os.path.join(
    'videos', f'MUSE-Size_X-{XDMG:.0f}_Y-{YDMG:.0f}.mp4'
)


def _dynamic_segments(m) -> tuple:
    """Damage-wall / invisible-wall line coords for the current medium map
    ``m`` (as returned by gen_MUSE_dmg)."""
    dmg_lines, inv_lines = [], []
    seen: set = set()
    for med in m.mediums.values():
        for obj in med.objs:
            if not isinstance(obj, (Segment, Ellipse)) or id(obj) in seen:
                continue
            seen.add(id(obj))
            if isinstance(obj, Ellipse):
                # sample the ellipse outline as a closed polyline
                psi = np.linspace(0., 2 * np.pi, 73)
                loc = np.column_stack([obj.a * np.cos(psi), obj.b * np.sin(psi)])
                rot = np.array([[obj._cos, -obj._sin], [obj._sin, obj._cos]])
                xy = loc.dot(rot.T) + obj.c
                dmg_lines.extend(zip(map(tuple, xy[:-1]), map(tuple, xy[1:])))
                continue
            pts = ((obj.a1[0], obj.a1[1]), (obj.a2[0], obj.a2[1]))
            if obj.color == 'blue':
                dmg_lines.append(pts)
            elif obj.color == 'white':
                inv_lines.append(pts)
    return dmg_lines, inv_lines


def build_frames() -> list:
    """Precompute per-size frame data once, so the animation callback only
    has to update artist data."""
    frames = []
    for size in SIZES:
        m, _ = gen_MUSE_dmg(xdmg=XDMG, ydmg=YDMG, xldmg=size, yldmg=size)
        dmg_lines, inv_lines = _dynamic_segments(m)
        frames.append(dict(size=size, dmg=dmg_lines, inv=inv_lines))
    return frames


def make_video(frames: list) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-10, PLATE_L + 10)
    ax.set_ylim(-10, PLATE_L + 10)
    ax.set_aspect('equal')
    ax.set_xlabel(r'\textit{X} [mm]')
    ax.set_ylabel(r'\textit{Y} [mm]')

    # Static elements: plate boundary + PZT sensors are identical every frame.
    ax.add_patch(plt.Rectangle((0, 0), PLATE_L, PLATE_L, fill=False,
                                edgecolor='black', linewidth=1.2))
    for p in pzt_pos:
        ax.add_patch(plt.Circle(p, r_pzt, color='black', fill=True, zorder=3))

    dmg_lc = LineCollection([], colors='red', linewidths=2.0, zorder=4)
    inv_lc = LineCollection([], colors='black', linestyles='--', linewidths=1.2, zorder=2)
    ax.add_collection(dmg_lc)
    ax.add_collection(inv_lc)

    def update(i):
        fr = frames[i]
        dmg_lc.set_segments(fr['dmg'])
        inv_lc.set_segments(fr['inv'])
        return dmg_lc, inv_lc

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    try:
        writer = animation.FFMpegWriter(
            fps=FPS,
            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'],
        )
        anim.save(OUTPUT, writer=writer, dpi=150)
        print(f'MP4 saved -> {OUTPUT}')
    except Exception as e:
        print(f'ffmpeg not available ({e}). Falling back to GIF...')
        gif_path = OUTPUT.replace('.mp4', '.gif')
        anim.save(gif_path, writer='pillow', fps=FPS)
        print(f'GIF saved -> {gif_path}')

    plt.close(fig)


if __name__ == '__main__':
    print(f'{len(SIZES)} sizes to render (from {SIZE_MIN} to {SIZE_MAX} mm, step {SIZE_STEP} mm)...')
    frames = build_frames()
    make_video(frames)

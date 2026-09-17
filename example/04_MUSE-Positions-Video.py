"""
MUSE damage-position sweep video.

Sweeps a 24x24 mm damage zone through the grid of positions the study was
run on (same formula as the original GLW/07_Validate-Mesher.py and
GLW/08_Execute-Damage_Simulation.py: gap sized to the largest studied
damage, 48 mm, so a single position grid is shared across all damage
sizes; step 16 mm; one plate quadrant, mirrored by symmetry for the rest),
and renders the resulting gen_MUSE_dmg() mesh at each position.

gen_MUSE_dmg() picks a vertical or horizontal 5-medium mesh depending on
whether the invisible internal walls would cross a PZT sensor (see
MUSE_dmg.py). This video makes that switch visible: the damage box is
drawn in red, invisible walls as dashed black lines (vertical pair above/
below the damage vs. horizontal pair left/right of it). A handful of
positions are geometrically invalid (damage wall intersects a sensor);
those frames are shown with no damage box.
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
# Study position grid
# ---------------------------------------------------------------------------
DMG_SIZE     = 28.    # mm, square damage side rendered in this video
DMG_SIZE_MAX = 48.    # mm, size used by the study to set the grid gap
GAP          = 11.
STEP         = 16.

x0 = GAP + DMG_SIZE_MAX / 2
y0 = GAP + DMG_SIZE_MAX / 2
_xs = np.linspace(x0, PLATE_L - x0, int((PLATE_L - 2 * x0) / STEP) + 1)
_xs = _xs[:int(_xs.shape[0] / 2)]
_ys = np.linspace(y0, PLATE_L - y0, int((PLATE_L - 2 * y0) / STEP) + 1)
_ys = _xs[:int(_ys.shape[0] / 2)]
POSITIONS = np.array(np.meshgrid(_xs, _ys)).T.reshape(-1, 2)

# ---------------------------------------------------------------------------
# Video parameters
# ---------------------------------------------------------------------------
FRAME_SECONDS = 0.1
FPS           = 1. / FRAME_SECONDS
OUTPUT        = os.path.join('videos', f'MUSE-Positions_D-{DMG_SIZE:.0f}.mp4')


def _dynamic_segments(m) -> tuple:
    """Damage-wall / invisible-wall line coords and mesh-layout label for
    the current medium map ``m`` (as returned by gen_MUSE_dmg)."""
    dmg_lines, inv_lines = [], []
    layout = None
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
                if layout is None:
                    (x1, y1), (x2, y2) = pts
                    layout = 'Vertical mesh' if x1 == x2 else 'Horizontal mesh'
    if layout is None:
        layout = 'Hole mesh'  # no invisible walls (mesh='holes')
    return dmg_lines, inv_lines, layout


def build_frames() -> list:
    """Precompute per-position frame data (segments + status) once, so the
    animation callback only has to update artist data."""
    frames = []
    for x, y in POSITIONS:
        try:
            m, _ = gen_MUSE_dmg(xdmg=x, ydmg=y, xldmg=DMG_SIZE, yldmg=DMG_SIZE)
        except ValueError as e:
            frames.append(dict(x=x, y=y, ok=False, error=str(e)))
            continue
        dmg_lines, inv_lines, layout = _dynamic_segments(m)
        frames.append(dict(x=x, y=y, ok=True, dmg=dmg_lines, inv=inv_lines, layout=layout))
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
    for i, p in enumerate(pzt_pos):
        ax.add_patch(plt.Circle(p, r_pzt, color='black', fill=True, zorder=3))
        ax.annotate(f'PZT{i + 1}', p, textcoords='offset points',
                    xytext=(0, 6), ha='center', fontsize=7)

    dmg_lc = LineCollection([], colors='red', linewidths=2.0, zorder=4)
    inv_lc = LineCollection([], colors='black', linestyles='--', linewidths=1.2, zorder=2)
    ax.add_collection(dmg_lc)
    ax.add_collection(inv_lc)

    def update(i):
        fr = frames[i]
        if fr['ok']:
            dmg_lc.set_segments(fr['dmg'])
            inv_lc.set_segments(fr['inv'])
        else:
            dmg_lc.set_segments([])
            inv_lc.set_segments([])
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
    print(f'{len(POSITIONS)} positions to render...')
    frames = build_frames()
    n_skipped = sum(1 for f in frames if not f['ok'])
    print(f'{n_skipped} position(s) skipped (sensor overlap).')
    make_video(frames)

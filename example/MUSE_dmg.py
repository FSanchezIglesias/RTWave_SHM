import os
import sys

import numpy as np

# Allow running from anywhere: make sure the repo root (parent of this
# example/ folder) is importable as the top-level `geom`/`RayTracing`
# packages, and that this folder itself is importable for `wavespeed`.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from geom.objects_2d import Segment, medium
from RayTracing.Sensors import Sensor
from geom.map_2d import Map2D
from geom.geom_utils import circunf_seg_intersect_2d, seg_seg_intersect_2d
from wavespeed import wavespeed_composite
from plate_config import (
    L, TH, R_PZT, BL, PZT_POS, WAVESPEED_HDF5,
    XDMG, YDMG, XLDMG, YLDMG, THDMG, RDMG, BLDMG, RMDMG,
)


# --- Constants (sourced from plate_config.py) ---
ws = wavespeed_composite(WAVESPEED_HDF5)
th = TH  # mm

r_pzt = R_PZT
bl = BL

l = L
pzt_pos = PZT_POS

xdmg = XDMG
xldmg = XLDMG

ydmg = YDMG
yldmg = YLDMG

thdmg = THDMG
rdmg = RDMG
bldmg = BLDMG
rmdmg = RMDMG


def _seg_intersects_sensor(seg: Segment, sensor: Sensor) -> bool:
    """Return True if *seg* intersects any boundary of *sensor*."""
    for b in sensor.bounds:
        if hasattr(b, 'c'):  # _CircBoundary
            if circunf_seg_intersect_2d(b.c, b.r, seg.a1, seg.a2):
                return True
        else:  # _StrBoundary
            if seg_seg_intersect_2d(seg.a1, seg.a2, b.a1, b.a2) is not None:
                return True
    return False


def _any_intersect(segs: list, sensors: list) -> bool:
    """Return True if any segment intersects any sensor boundary."""
    return any(_seg_intersects_sensor(s, pzt) for s in segs for pzt in sensors)


def _build_vertical(
    xdmg: float, xldmg: float, ydmg: float, yldmg: float,
    l: float, bl: float, rdmg: float, bldmg: float, rmdmg: float,
) -> tuple:
    """Build 5-element vertical mesh (big left/right elements).

    Layout::

        ┌─────┬──────┬─────┐
        │     │ top  │     │
        │ lft ├──────┤ rgt │
        │     │ dmg  │     │
        │     ├──────┤     │
        │     │ bot  │     │
        └─────┴──────┴─────┘

    Invisible walls are the 4 vertical segments at x=xl and x=xr,
    above and below the damage zone.

    Returns:
        (medium_objs, dmg_segs, inv_segs) where medium_objs is a list
        of 5 object-lists: [lft, bot, dmg, top, rgt].
    """
    xl = xdmg - xldmg / 2
    xr = xdmg + xldmg / 2
    yb = ydmg - yldmg / 2
    yt = ydmg + yldmg / 2

    P1  = np.array([0.,  0.])
    P2  = np.array([xl,  0.])
    P3  = np.array([xr,  0.])
    P4  = np.array([l,   0.])
    P5  = np.array([xl,  yb])
    P6  = np.array([xr,  yb])
    P7  = np.array([xl,  yt])
    P8  = np.array([xr,  yt])
    P9  = np.array([0.,  l])
    P10 = np.array([xl,  l])
    P11 = np.array([xr,  l])
    P12 = np.array([l,   l])

    kw_bl  = dict(boundary_losses=bl,    ratio_mode=1.)
    kw_dmg = dict(boundary_losses=bldmg, ratio_rfl=rdmg, ratio_mode=rmdmg, color='blue')
    kw_inv = dict(boundary_losses=0.,    ratio_rfl=0.,   ratio_mode=1.,    color='white')

    # Plate boundaries
    S_b1  = Segment(P1,  P2,  **kw_bl)
    S_b2  = Segment(P2,  P3,  **kw_bl)
    S_b3  = Segment(P3,  P4,  **kw_bl)
    S_lft = Segment(P1,  P9,  **kw_bl)
    S_rgt = Segment(P4,  P12, **kw_bl)
    S_t1  = Segment(P9,  P10, **kw_bl)
    S_t2  = Segment(P10, P11, **kw_bl)
    S_t3  = Segment(P11, P12, **kw_bl)

    # Damage walls
    S_dmg_bot = Segment(P5, P6, **kw_dmg)
    S_dmg_lft = Segment(P5, P7, **kw_dmg)
    S_dmg_rgt = Segment(P6, P8, **kw_dmg)
    S_dmg_top = Segment(P7, P8, **kw_dmg)

    # Invisible internal walls (vertical, outside damage zone)
    S_inv_lb = Segment(P2, P5,  **kw_inv)  # left side, below damage
    S_inv_rb = Segment(P3, P6,  **kw_inv)  # right side, below damage
    S_inv_lt = Segment(P7, P10, **kw_inv)  # left side, above damage
    S_inv_rt = Segment(P8, P11, **kw_inv)  # right side, above damage

    medium_objs = [
        [S_lft, S_b1, S_inv_lb, S_dmg_lft, S_inv_lt, S_t1],   # lft
        [S_b2, S_inv_rb, S_dmg_bot, S_inv_lb],                  # bot
        [S_dmg_bot, S_dmg_lft, S_dmg_rgt, S_dmg_top],           # dmg
        [S_dmg_top, S_inv_rt, S_t2, S_inv_lt],                  # top
        [S_b3, S_rgt, S_t3, S_inv_rt, S_dmg_rgt, S_inv_rb],    # rgt
    ]
    dmg_segs = [S_dmg_bot, S_dmg_lft, S_dmg_rgt, S_dmg_top]
    inv_segs = [S_inv_lb, S_inv_rb, S_inv_lt, S_inv_rt]

    return medium_objs, dmg_segs, inv_segs


def _build_horizontal(
    xdmg: float, xldmg: float, ydmg: float, yldmg: float,
    l: float, bl: float, rdmg: float, bldmg: float, rmdmg: float,
) -> tuple:
    """Build 5-element horizontal mesh (big top/bottom elements).

    Layout::

        ┌───────────────────┐
        │    top (wide)     │
        ├──────┬─────┬──────┤
        │ lft  │ dmg │ rgt  │
        ├──────┴─────┴──────┤
        │    bot (wide)     │
        └───────────────────┘

    Invisible walls are the 4 horizontal segments at y=yb and y=yt,
    left and right of the damage zone.

    Returns:
        (medium_objs, dmg_segs, inv_segs) where medium_objs is a list
        of 5 object-lists: [bot, lft, dmg, rgt, top].
    """
    xl = xdmg - xldmg / 2
    xr = xdmg + xldmg / 2
    yb = ydmg - yldmg / 2
    yt = ydmg + yldmg / 2

    P1  = np.array([0.,  0.])
    P2  = np.array([0.,  yb])
    P3  = np.array([0.,  yt])
    P4  = np.array([0.,  l])
    P5  = np.array([xl,  yb])
    P6  = np.array([xl,  yt])
    P7  = np.array([xr,  yb])
    P8  = np.array([xr,  yt])
    P9  = np.array([l,   0.])
    P10 = np.array([l,   yb])
    P11 = np.array([l,   yt])
    P12 = np.array([l,   l])

    kw_bl  = dict(boundary_losses=bl,    ratio_mode=1.)
    kw_dmg = dict(boundary_losses=bldmg, ratio_rfl=rdmg, ratio_mode=rmdmg, color='blue')
    kw_inv = dict(boundary_losses=0.,    ratio_rfl=0.,   ratio_mode=1.,    color='white')

    # Plate boundaries
    S_b_full = Segment(P1,  P9,  **kw_bl)  # bottom plate (full width)
    S_lft_b  = Segment(P1,  P2,  **kw_bl)  # left plate, below damage
    S_lft_c  = Segment(P2,  P3,  **kw_bl)  # left plate, beside damage
    S_lft_t  = Segment(P3,  P4,  **kw_bl)  # left plate, above damage
    S_rgt_b  = Segment(P9,  P10, **kw_bl)  # right plate, below damage
    S_rgt_c  = Segment(P10, P11, **kw_bl)  # right plate, beside damage
    S_rgt_t  = Segment(P11, P12, **kw_bl)  # right plate, above damage
    S_t_full = Segment(P4,  P12, **kw_bl)  # top plate (full width)

    # Damage walls
    S_dmg_bot = Segment(P5, P7, **kw_dmg)
    S_dmg_lft = Segment(P5, P6, **kw_dmg)
    S_dmg_rgt = Segment(P7, P8, **kw_dmg)
    S_dmg_top = Segment(P6, P8, **kw_dmg)

    # Invisible internal walls (horizontal, outside damage zone)
    S_inv_bl = Segment(P2, P5,  **kw_inv)  # bottom side, left of damage
    S_inv_br = Segment(P7, P10, **kw_inv)  # bottom side, right of damage
    S_inv_tl = Segment(P3, P6,  **kw_inv)  # top side, left of damage
    S_inv_tr = Segment(P8, P11, **kw_inv)  # top side, right of damage

    medium_objs = [
        [S_b_full, S_lft_b, S_inv_bl, S_dmg_bot, S_inv_br, S_rgt_b],  # bot
        [S_lft_c, S_inv_bl, S_dmg_lft, S_inv_tl],                      # lft
        [S_dmg_bot, S_dmg_lft, S_dmg_rgt, S_dmg_top],                  # dmg
        [S_inv_br, S_rgt_c, S_inv_tr, S_dmg_rgt],                      # rgt
        [S_lft_t, S_t_full, S_rgt_t, S_inv_tl, S_dmg_top, S_inv_tr],  # top
    ]
    dmg_segs = [S_dmg_bot, S_dmg_lft, S_dmg_rgt, S_dmg_top]
    inv_segs = [S_inv_bl, S_inv_br, S_inv_tl, S_inv_tr]

    return medium_objs, dmg_segs, inv_segs


def gen_MUSE_dmg(ws=ws, r_pzt=r_pzt, bl=bl, l=l, th=th, pzt_pos=pzt_pos,
                 xdmg=xdmg, xldmg=xldmg, ydmg=ydmg, yldmg=yldmg,
                 wsdmg=ws, rdmg=rdmg, bldmg=bldmg, rmdmg=rmdmg, thdmg=thdmg):

    pzts = [Sensor('circ', [p, r_pzt], name='PZT{}'.format(i + 1))
            for i, p in enumerate(pzt_pos)]

    # Try vertical mesh first; fall back to horizontal if invisible walls
    # intersect any sensor circumference.
    medium_objs, dmg_segs, inv_segs = _build_vertical(
        xdmg, xldmg, ydmg, yldmg, l, bl, rdmg, bldmg, rmdmg
    )
    if _any_intersect(inv_segs, pzts):
        medium_objs, dmg_segs, inv_segs = _build_horizontal(
            xdmg, xldmg, ydmg, yldmg, l, bl, rdmg, bldmg, rmdmg
        )
        if _any_intersect(inv_segs, pzts):
            raise ValueError(
                'Both vertical and horizontal mesh layouts produce '
                'invisible walls that intersect a sensor circumference. '
                'Reposition the damage or the sensors.'
            )

    if _any_intersect(dmg_segs, pzts):
        raise ValueError(
            'A damage wall intersects a sensor circumference. '
            'Reposition the damage or the sensors.'
        )

    # Index 2 is always the damage medium across both layouts.
    mediums = [
        medium(ws,    th,    xi=1.e-3),
        medium(ws,    th,    xi=1.e-3),
        medium(wsdmg, thdmg, xi=5.e-3),
        medium(ws,    th,    xi=1.e-3),
        medium(ws,    th,    xi=1.e-3),
    ]
    for m, objs in zip(mediums, medium_objs):
        m.add_objs(objs)

    map_ = Map2D(mediums=mediums, background=False)
    for pzt in pzts:
        try:
            map_.add_sensor(pzt)
        except KeyError as e:
            print(e)

    return map_, pzts


def gen_MUSE_intact(ws=ws, r_pzt=r_pzt, bl=bl, l=l, pzt_pos=pzt_pos):

    P01 = np.array([0., 0.])
    P02 = np.array([l, 0.])
    P03 = np.array([l, l])
    P04 = np.array([0., l])

    # Boundaries
    S01 = Segment(P01, P02, boundary_losses=bl, ratio_mode=1.)
    S02 = Segment(P02, P03, boundary_losses=bl, ratio_mode=1.)
    S03 = Segment(P03, P04, boundary_losses=bl, ratio_mode=1.)
    S04 = Segment(P04, P01, boundary_losses=bl, ratio_mode=1.)

    # Medium initialization
    m1 = medium(ws, th, xi=1.e-3)

    m1.add_objs([S01, S02, S03, S04])

    pzts = [Sensor('circ', [p, r_pzt], name='PZT{}'.format(i + 1))
            for i, p in enumerate(pzt_pos)]
    mediums = [m1, ]

    m = Map2D(mediums=mediums, background=True)
    for k in pzts:
        m.add_sensor(k)

    return m, pzts

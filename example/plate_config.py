"""Shared plate, PZT-grid, damage and simulation configuration for the MUSE examples.

`MUSE_dmg.py` (geometry generator), `Run-Single-Damage.py` and
`plot_wave_video.py` all import their plate/damage parameters from here, so
the layout is defined in exactly one place instead of being duplicated (and
drifting) across scripts.
"""
import os

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Plate / laminate -------------------------------------------------
L = 726.                          # plate side length [mm]
TH = 1.288                        # plate thickness [mm]
STACKING = '(+45, -45, 90, 0)$'
MATERIAL = 'AS4/8552'
WAVESPEED_HDF5 = os.path.join(_THIS_DIR, 'muse_stacking.hdf5')

# --- PZT sensor grid ----------------------------------------------------
R_PZT = 4.                         # PZT sensor radius [mm]
BL = 0.25                          # plate-boundary loss ratio
PZT_POS = [
    [181.5, 145.2],
    [181.5, 290.4],
    [181.5, 435.6],
    [181.5, 580.8],
    [544.5, 145.2],
    [544.5, 290.4],
    [544.5, 435.6],
    [544.5, 580.8],
]

# --- Damage ---------------------------------------------------------------
XDMG, YDMG = L / 2, 149.           # midway between the two PZT columns (x=181.5, x=544.5)
XLDMG, YLDMG = 12., 12.
THDMG = 3.
RDMG = 0.1
BLDMG = 0.05
RMDMG = 1

# --- Simulation / source ---------------------------------------------------
NRAYS = 2_001
F = 350.e3                                    # burst centre frequency [Hz]
T = np.linspace(0., 0.0002, 10_000)           # time vector [s] (50 MHz)
SOURCE_IDX = 0                                # PZT1 as source

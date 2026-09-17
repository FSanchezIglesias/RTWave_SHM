"""
Simulate and plot sensor signals for the intact and damaged plate.

Intact results  → results/intact_350k.hdf5
Damaged results → results/dmg20_P1P5_350k.hdf5  (20 mm damage mid PZT1–PZT5)

Both simulation caches are filled on first run and reused on subsequent runs.
Each output figure overlays intact (dark) and damaged (red) waveforms so the
change introduced by the damage can be read directly.
"""

import gc
import os

import h5py
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from MUSE_dmg import gen_MUSE_intact, gen_MUSE_dmg
from RTWave_SHM.RayTracing.Ray import Beam_from_pzt

# ---------------------------------------------------------------------------
# Parameters — simulation
# ---------------------------------------------------------------------------
F     = 350.e3
T     = np.linspace(0., 0.0002, 10_000)   # 50 MHz
NRAYS = 2_001
NFFT  = 500

# Intact
INTACT_HDF5 = os.path.join('results', 'intact_350k.hdf5')
INTACT_KEY  = 'intact'

# Damage: 20 mm square at the centre of the 726×726 mm plate
XDMG, YDMG   = 363.0, 363.0
XLDMG, YLDMG = 20.,   20.
THDMG         = 3.
RDMG          = 0.1
BLDMG         = 0.05
RMDMG         = 1.
DMG_HDF5      = os.path.join('results', 'dmg20_center_350k.hdf5')
DMG_KEY       = f'dmg20_{XDMG:.1f}_{YDMG:.1f}'

# ---------------------------------------------------------------------------
# Parameters — figure
# ---------------------------------------------------------------------------
OUT_DIR      = 'figures'
OUT_TEMPLATE      = os.path.join(OUT_DIR, 'signals_PZT{source}.pdf')
OUT_DIFF_TEMPLATE      = os.path.join(OUT_DIR, 'signals_diff_PZT{source}.pdf')
OUT_DIFF_OVL_TEMPLATE   = os.path.join(OUT_DIR, 'signals_diff_overlay_PZT{source}.pdf')
OUT_INTACT_OVL_TEMPLATE = os.path.join(OUT_DIR, 'signals_intact_overlay_PZT{source}.pdf')
USE_LATEX    = True

COLOR_INTACT = '#1a1a2e'
COLOR_DMG    = '#c0392b'
OFFSET_SCALE = 2.2   # vertical gap between stacked traces (normalised units)
AMP_SCALE    = 2.5   # amplitude multiplier — increase to make waveforms taller

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": USE_LATEX,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
    "axes.labelsize": 11,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ---------------------------------------------------------------------------
# Helper — run one simulation and cache all 8 sources to an HDF5 file
# ---------------------------------------------------------------------------
def _run_and_cache(hdf5_file: str, key_prefix: str, build_map) -> None:
    """Simulate all 8 sources and save to *hdf5_file* under *key_prefix*/PZTn."""
    os.makedirs('results', exist_ok=True)
    with h5py.File(hdf5_file, 'a') as h5f:
        if 'time' not in h5f:
            h5f['time'] = T
            h5f.attrs['Stacking'] = '(+45, -45, 90, 0)'
            h5f.attrs['Material'] = 'AS4/8552'
            h5f.attrs['Plate dimensions'] = [726., 726.]

    for source in range(8):
        key = f'{key_prefix}/PZT{source + 1}'
        with h5py.File(hdf5_file, 'r') as h5f:
            if key in h5f:
                print(f'  PZT{source + 1}: cached — skipping.')
                continue

        print(f'  PZT{source + 1}: building geometry...')
        m, pzts = build_map()

        print(f'  PZT{source + 1}: tracing {NRAYS} rays...')
        ibeam = Beam_from_pzt(
            NRAYS, pzts[source], power=NRAYS / 8,
            f=F, npeaks=3, nfft=NFFT, t=T,
        )
        m.set_init_beam(ibeam)
        m.calc_t()

        print(f'  PZT{source + 1}: computing signals...')
        m.calc_signal()
        m.save_signals(hdf5_file, key)
        m.close_h5()
        gc.collect()
        print(f'  PZT{source + 1}: done.')


# ---------------------------------------------------------------------------
# Helper — load all sources from an HDF5 file
# ---------------------------------------------------------------------------
def _load(hdf5_file: str, key_prefix: str) -> tuple[np.ndarray, dict]:
    with h5py.File(hdf5_file, 'r') as h5f:
        t = h5f['time'][:]
        data = {}
        for source in range(8):
            key = f'{key_prefix}/PZT{source + 1}'
            if key in h5f:
                data[source] = {
                    'data':    h5f[key][:],
                    'columns': list(h5f[key].attrs['columns']),
                }
    return t, data


# ---------------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------------
print('=== Intact plate ===')
_run_and_cache(INTACT_HDF5, INTACT_KEY, gen_MUSE_intact)

print('=== Damaged plate (20 mm, PZT1–PZT5 midpoint) ===')
_run_and_cache(
    DMG_HDF5, DMG_KEY,
    lambda: gen_MUSE_dmg(
        xdmg=XDMG, ydmg=YDMG, xldmg=XLDMG, yldmg=YLDMG,
        thdmg=THDMG, rdmg=RDMG, bldmg=BLDMG, rmdmg=RMDMG,
    ),
)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
t, intact_signals = _load(INTACT_HDF5, INTACT_KEY)
_,  dmg_signals   = _load(DMG_HDF5,    DMG_KEY)

if not intact_signals:
    raise RuntimeError(f'No intact signals found in {INTACT_HDF5}.')
if not dmg_signals:
    raise RuntimeError(f'No damaged signals found in {DMG_HDF5}.')

# ---------------------------------------------------------------------------
# Plot — one figure per source, intact + damaged overlaid
# ---------------------------------------------------------------------------
t_us = t * 1.e6   # s → µs

os.makedirs(OUT_DIR, exist_ok=True)


def _zero_idx(name: str) -> str:
    """'PZT3' → 'PZT2'  (shift from 1-based to 0-based numbering)."""
    return f'PZT{int(name[3:]) - 1}'

legend_handles = [
    mlines.Line2D([], [], color=COLOR_INTACT, lw=0.8, label='Intact'),
    mlines.Line2D([], [], color=COLOR_DMG,    lw=0.8, label='Damaged (20 mm)'),
]

for source in range(8):
    if source not in intact_signals or source not in dmg_signals:
        continue

    info_i   = intact_signals[source]
    info_d   = dmg_signals[source]
    data_i   = info_i['data']
    data_d   = info_d['data']
    columns  = info_i['columns']
    labels   = [_zero_idx(c) for c in columns]   # 0-based display names
    n_rx     = data_i.shape[1]
    src_label = f'PZT{source}'                    # 0-based source label

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, col in enumerate(columns):
        offset = i * OFFSET_SCALE
        ax.plot(t_us, AMP_SCALE * data_i[:, i] + offset,
                lw=0.75, color=COLOR_INTACT, alpha=0.9)
        ax.plot(t_us, AMP_SCALE * data_d[:, i] + offset,
                lw=0.75, color=COLOR_DMG,    alpha=0.85, linestyle='--')

    # Receiver labels on the right axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([i * OFFSET_SCALE for i in range(n_rx)])
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.tick_params(length=0)

    ax.set_yticks([])
    ax.set_xlabel(r'Time [$\mu$s]')
    # ax.set_title(
    #     f'Source PZT{source + 1} — intact vs. damaged\n'
    #     f'AS4/8552 (+45,−45,90,0)$_s$,  f = 350 kHz  |  '
    #     f'damage 20 mm @ ({XDMG:.0f}, {YDMG:.0f}) mm',
    #     fontsize=9,
    # )
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.85)

    plt.tight_layout()
    out = OUT_TEMPLATE.format(source=source)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Figure saved → {out}')
    plt.show()

    # --- Difference figure (damaged − intact) ---
    COLOR_DIFF = '#2980b9'

    fig_d, ax_d = plt.subplots(figsize=(8, 5))

    for i, col in enumerate(columns):
        offset = i * OFFSET_SCALE
        diff = AMP_SCALE * (data_d[:, i] - data_i[:, i])
        ax_d.plot(t_us, diff + offset, lw=0.8, color=COLOR_DIFF, alpha=0.9)

    ax2_d = ax_d.twinx()
    ax2_d.set_ylim(ax_d.get_ylim())
    ax2_d.set_yticks([i * OFFSET_SCALE for i in range(n_rx)])
    ax2_d.set_yticklabels(labels, fontsize=8)
    ax2_d.tick_params(length=0)

    ax_d.axhline(0, color='black', lw=0.4, alpha=0.4)
    ax_d.set_yticks([])
    ax_d.set_xlabel(r'Time [$\mu$s]')

    plt.tight_layout()
    out_d = OUT_DIFF_TEMPLATE.format(source=source)
    plt.savefig(out_d, dpi=150, bbox_inches='tight')
    print(f'Figure saved → {out_d}')
    plt.show()

    # --- Difference overlay — all channels share the same origin, Spectral_r ---
    cmap   = plt.get_cmap('Spectral_r')
    colors = [cmap(i / (n_rx - 1)) for i in range(n_rx)]

    fig_o, ax_o = plt.subplots(figsize=(6.4, 4.8))

    for i, col in enumerate(columns):
        diff = AMP_SCALE * (data_d[:, i] - data_i[:, i])
        ax_o.plot(t_us, diff, lw=0.9, color=colors[i], alpha=0.9, label=labels[i])

    ax_o.axhline(0, color='black', lw=0.5, alpha=0.4)
    ax_o.set_xlabel(r'\textit{Time} [$\mu$s]', fontsize=13)
    ax_o.set_ylabel(r'\textit{Amplitude difference} [V]', fontsize=13)
    ax_o.tick_params(labelsize=12)
    ax_o.legend(
        loc='upper left', ncol=2, fontsize=11, framealpha=1, edgecolor='black'
    )

    plt.grid(linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    out_o = OUT_DIFF_OVL_TEMPLATE.format(source=source)
    plt.savefig(out_o, dpi=150, bbox_inches='tight')
    print(f'Figure saved → {out_o}')
    plt.show()

    # --- Intact overlay — same layout as diff overlay ---
    fig_oi, ax_oi = plt.subplots(figsize=(6.4, 4.8))

    for i, col in enumerate(columns):
        ax_oi.plot(t_us, AMP_SCALE * data_i[:, i],
                   lw=0.9, color=colors[i], alpha=0.9, label=labels[i])

    ax_oi.axhline(0, color='black', lw=0.5, alpha=0.4)
    ax_oi.set_xlabel(r'\textit{Time} [$\mu$s]', fontsize=13)
    ax_oi.set_ylabel(r'\textit{Amplitude} [V]', fontsize=13)
    ax_oi.tick_params(labelsize=12)
    ax_oi.legend(
        loc='upper right', ncol=2, fontsize=11, framealpha=1, edgecolor='black'
    )

    plt.grid(linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    out_oi = OUT_INTACT_OVL_TEMPLATE.format(source=source)
    plt.savefig(out_oi, dpi=150, bbox_inches='tight')
    print(f'Figure saved → {out_oi}')
    plt.show()

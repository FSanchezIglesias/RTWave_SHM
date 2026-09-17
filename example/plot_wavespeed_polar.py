"""
Plot S0 and A0 wave speeds as a function of propagation angle
for the AS4/8552 (±45, 90, 0) composite laminate.
"""
import numpy as np
import matplotlib.pyplot as plt
from wavespeed import wavespeed_composite

# --- Parameters (matching 02_Run-Single-Damage.py) ---
WS_FILE = 'muse_stacking.hdf5'
F = 350.e3      # Hz
TH = 1.288      # mm (intact plate)
F_D = F * TH / 1.e6  # MHz·mm  →  0.4508

# --- Load dispersion data ---
ws = wavespeed_composite(WS_FILE)

# --- Angular sweep [0, 2π] ---
# The medium uses theta % pi (material symmetry), so we mirror [0,π] → [0,2π]
theta_half = np.linspace(0., np.pi, 360)
theta_full = np.concatenate([theta_half, theta_half + np.pi])

v_s0 = np.array([ws.S0((F_D, t % np.pi)) * 1.e3 for t in theta_full])  # m/s
v_a0 = np.array([ws.A0((F_D, t % np.pi)) * 1.e3 for t in theta_full])  # m/s

# --- Figure: polar + Cartesian ---
fig = plt.figure(figsize=(14, 6))
fig.suptitle(
    f'Guided Lamb Wave speeds — AS4/8552 (+45,−45,90,0)\n'
    f'f = {F/1e3:.0f} kHz,  th = {TH} mm  '
    f'(f·h = {F_D:.4f} MHz·mm)',
    fontsize=12,
)

# ── Polar plot ──────────────────────────────────────────────
ax_pol = fig.add_subplot(1, 2, 1, projection='polar')

ax_pol.plot(theta_full, v_s0 / 1e3, color='tab:blue',  lw=2, label='S0')
ax_pol.plot(theta_full, v_a0 / 1e3, color='tab:orange', lw=2, label='A0')

ax_pol.set_title('Phase velocity [km/s]', pad=15, fontsize=10)
ax_pol.set_theta_zero_location('E')   # 0° points right
ax_pol.set_theta_direction(1)         # counter-clockwise
ax_pol.legend(loc='lower right', bbox_to_anchor=(1.25, -0.05))

# ── Cartesian plot ───────────────────────────────────────────
ax_cart = fig.add_subplot(1, 2, 2)

theta_deg = np.degrees(theta_full)
ax_cart.plot(theta_deg, v_s0 / 1e3, color='tab:blue',   lw=2, label='S0')
ax_cart.plot(theta_deg, v_a0 / 1e3, color='tab:orange',  lw=2, label='A0')

ax_cart.set_xlabel('Propagation angle [°]')
ax_cart.set_ylabel('Phase velocity [km/s]')
ax_cart.set_xlim(0, 360)
ax_cart.set_xticks(np.arange(0, 361, 45))
ax_cart.grid(True, alpha=0.4)
ax_cart.legend()

# Annotate min/max of S0
s0_max_idx = np.argmax(v_s0)
s0_min_idx = np.argmin(v_s0)
ax_cart.annotate(
    f'{v_s0[s0_max_idx]/1e3:.2f} km/s',
    xy=(theta_deg[s0_max_idx], v_s0[s0_max_idx]/1e3),
    xytext=(10, 5), textcoords='offset points', color='tab:blue', fontsize=8
)
ax_cart.annotate(
    f'{v_s0[s0_min_idx]/1e3:.2f} km/s',
    xy=(theta_deg[s0_min_idx], v_s0[s0_min_idx]/1e3),
    xytext=(10, -12), textcoords='offset points', color='tab:blue', fontsize=8
)

# Annotate min/max of A0
a0_max_idx = np.argmax(v_a0)
a0_min_idx = np.argmin(v_a0)
ax_cart.annotate(
    f'{v_a0[a0_max_idx]/1e3:.2f} km/s',
    xy=(theta_deg[a0_max_idx], v_a0[a0_max_idx]/1e3),
    xytext=(10, 5), textcoords='offset points', color='tab:orange', fontsize=8
)
ax_cart.annotate(
    f'{v_a0[a0_min_idx]/1e3:.2f} km/s',
    xy=(theta_deg[a0_min_idx], v_a0[a0_min_idx]/1e3),
    xytext=(10, -12), textcoords='offset points', color='tab:orange', fontsize=8
)

plt.tight_layout()
# plt.savefig('wavespeed_polar.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: wavespeed_polar.png')

# --- Summary table ---
print(f'\n{"Mode":<6} {"Min [m/s]":>12} {"@ angle":>10} {"Max [m/s]":>12} {"@ angle":>10}')
print('-' * 55)
for name, v in [('S0', v_s0), ('A0', v_a0)]:
    print(
        f'{name:<6} {v.min():>12.1f} {theta_deg[v.argmin()]:>9.1f}°'
        f' {v.max():>12.1f} {theta_deg[v.argmax()]:>9.1f}°'
    )

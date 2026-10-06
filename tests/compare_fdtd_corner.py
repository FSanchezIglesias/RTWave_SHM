"""Reference for the penetrable corner: 2D scalar-wave FDTD vs ray tracer (GO / GO+UTD).

Run from the repository root (about 2 min)::

    python tests/compare_fdtd_corner.py [--out-dir tests/out]

Writes ``fdtd_corner.png`` and ``fdtd_corner.npz`` and prints the RMS
deviation of the ray model from the finite-difference reference, with and
without corner diffraction.  Results (2026-09-18): arc across the corner line
RMS 0.13 with diffraction vs 0.37 without; receiver lines behind the damage
0.38 vs 0.48 (the remaining deviation is the sharper ray caustics of the
focusing inside the slower square).

Idealised common problem: isotropic non-dispersive media, c = 5.205 mm/us in
the plate and 4.432 mm/us inside the 48 x 48 mm square at (355, 147), point
source at PZT1 (181.5, 145.2), 3-cycle Hann burst at 350 kHz, absorbing
outer edges, no damping.  Scalar impedance contrast (equal density):
R = (c1-c2)/(c1+c2) = 0.080, T = 1 - 0.08 = 0.92 -> ray walls ratio_rfl=0.08,
bl=0.  Receivers: vertical lines at x = 430 and 500 (y = 100..200) and an arc
of R = 270 mm around the source across the top-left corner line.
Metric: peak of the direct-arrival envelope, damaged / intact.
"""
import os, sys, time
import numpy as np
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
OUT = os.path.join(_REPO_ROOT, 'tests', 'out')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from RayTracing.Signal import burst_hann

C1, C2 = 5.205, 4.432          # mm/us
F = 0.35                        # MHz
SRC = np.array([181.5, 145.2])
XD, YD, LD = 355., 147., 48.
XL, XR, YB, YT = XD - LD / 2, XD + LD / 2, YD - LD / 2, YD + LD / 2
V1 = np.array([XL, YT])
ang_c = np.arctan2(V1[1] - SRC[1], V1[0] - SRC[0])

# receivers
lines = [(x, y) for x in (430., 500.) for y in np.arange(100., 200.1, 4.)]
arc = [(270. * np.cos(ang_c + np.deg2rad(da)) + SRC[0], 270. * np.sin(ang_c + np.deg2rad(da)) + SRC[1])
       for da in (-6., -4., -2., -1., 1., 2., 4., 6.)]
RECV = lines + arc


def peak_direct(sig, t, r):
    """Peak |signal| in the direct-arrival window (path r [mm])."""
    win = t < (r + 60.) / C1
    return np.abs(sig[win]).max()


# ------------------------------------------------------------------ FDTD
def fdtd(damaged, dx=0.5, tmax=115., x0=90., x1=570., y0=0., y1=310., sponge=30.):
    nx, ny = int((x1 - x0) / dx) + 1, int((y1 - y0) / dx) + 1
    xs = x0 + dx * np.arange(nx); ys = y0 + dx * np.arange(ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    c = np.full((nx, ny), C1, dtype=np.float32)
    if damaged:
        c[(X > XL) & (X < XR) & (Y > YB) & (Y < YT)] = C2
    dt = 0.4 * dx / C1
    nt = int(tmax / dt)
    t = dt * np.arange(nt)
    src = burst_hann(t * 1e-6, 1.0, F * 1e6, F * 1e6 / 3.)
    # sponge (exponential damping profile)
    d = np.minimum.reduce([X - x0, x1 - X, Y - y0, y1 - Y])
    damp = np.exp(-((np.clip(sponge - d, 0., None) / sponge) ** 2) * 0.15).astype(np.float32)
    coef = (c * dt / dx) ** 2
    u = np.zeros((nx, ny), np.float32); up = np.zeros_like(u)
    isx, isy = int(round((SRC[0] - x0) / dx)), int(round((SRC[1] - y0) / dx))
    ridx = [(int(round((x - x0) / dx)), int(round((y - y0) / dx))) for x, y in RECV]
    rec = np.zeros((len(RECV), nt), np.float32)
    t0 = time.perf_counter()
    for k in range(nt):
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4. * u[1:-1, 1:-1])
        un = 2. * u - up + coef * lap
        un[isx, isy] += src[k]
        un *= damp; u *= damp
        up, u = u, un
        for i, (ix, iy) in enumerate(ridx):
            rec[i, k] = u[ix, iy]
    print(f'FDTD damaged={damaged}: {nx}x{ny} cells, {nt} steps, {time.perf_counter()-t0:.0f} s')
    return t, rec


# ------------------------------------------------------------------ ray tracer
def rays(damaged, diffraction):
    import RayTracing.Ray as RM
    from RayTracing.Ray import Beam
    from RayTracing.Sensors import Sensor
    from geom.objects_2d import Segment, medium
    from geom.map_2d import Map2D
    RM.diffraction = diffraction

    class WS:
        def __init__(self, v): self.v = v
        def S0(self, fa): return self.v
        def A0(self, fa): return self.v
    L = 726.
    P = [np.array([0., 0.]), np.array([L, 0.]), np.array([L, L]), np.array([0., L])]
    kw_abs = dict(boundary_losses=1., ratio_rfl=1., ratio_mode=1.)
    plate = [Segment(P[i], P[(i + 1) % 4], **kw_abs) for i in range(4)]
    m1 = medium(WS(C1 * 1e3), 1., xi=1.e-12, dispersive=False)  # xi=0 mediums hash-collide
    meds = [m1]
    if damaged:
        kw = dict(boundary_losses=0., ratio_rfl=0.08, ratio_mode=1.)
        Q = [np.array([XL, YB]), np.array([XR, YB]), np.array([XL, YT]), np.array([XR, YT])]
        dmg = [Segment(Q[0], Q[1], **kw), Segment(Q[0], Q[2], **kw), Segment(Q[1], Q[3], **kw), Segment(Q[2], Q[3], **kw)]
        m2 = medium(WS(C2 * 1e3), 1., xi=2.e-12, dispersive=False)
        m1.add_objs(dmg + plate); m2.add_objs(dmg); meds.append(m2)
    else:
        m1.add_objs(plate)
    r_map = Map2D(mediums=meds)
    sens = []
    for x, y in RECV:
        s = Sensor('circ', [np.array([x, y]), 1.5], name=f'{x:.1f}_{y:.1f}')
        r_map.add_sensor(s); sens.append(s)
    T = np.linspace(0., 115e-6, 5750)
    beam = Beam(3001, [SRC.copy()], m1, f=F * 1e6, npeaks=3, nfft=300, t=T, kind='S0', power=3001.)
    r_map.set_init_beam(beam); r_map.calc_t()
    out = []
    for s in sens:
        s.signal()
        out.append(np.zeros(len(T)) if s.signal_s is None else s.signal_s)
    return T * 1e6, np.array(out)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=OUT)
    OUT = ap.parse_args().out_dir
    os.makedirs(OUT, exist_ok=True)
    res = {}
    tf, rec_i = fdtd(False); _, rec_d = fdtd(True)
    r = [np.hypot(x - SRC[0], y - SRC[1]) for x, y in RECV]
    res['FDTD'] = np.array([peak_direct(rec_d[i], tf, r[i]) / peak_direct(rec_i[i], tf, r[i]) for i in range(len(RECV))])
    np.savez(os.path.join(OUT, 'fdtd_corner_rec.npz'), t=tf, rec_i=rec_i, rec_d=rec_d)
    tr, ray_i = rays(False, False)
    for lab, diff in (('GO', False), ('GO+UTD', True)):
        _, ray_d = rays(True, diff)
        res[lab] = np.array([peak_direct(ray_d[i], tr, r[i]) / peak_direct(ray_i[i], tr, r[i]) for i in range(len(RECV))])
    np.savez(os.path.join(OUT, 'fdtd_corner.npz'), **res, recv=np.array(RECV))

    n_line = len(lines)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    ys = np.arange(100., 200.1, 4.)
    for k, x in enumerate((430., 500.)):
        ax = axs[k]; sl = slice(k * len(ys), (k + 1) * len(ys))
        for lab, st in (('FDTD', 'k-'), ('GO', 'o--'), ('GO+UTD', 's-')):
            ax.plot(ys, res[lab][sl], st, label=lab, lw=1.5 if lab == 'FDTD' else 1, ms=4)
        for yc in (YB, YT): ax.axvline(yc, color='gray', ls=':')
        ax.set_title(f'x = {x:.0f} mm'); ax.set_xlabel('y [mm]'); ax.set_ylabel('peak damaged / intact'); ax.grid(alpha=.3); ax.legend()
    ax = axs[2]; das = [-6, -4, -2, -1, 1, 2, 4, 6]
    for lab, st in (('FDTD', 'k-'), ('GO', 'o--'), ('GO+UTD', 's-')):
        ax.plot(das, res[lab][n_line:], st, label=lab, ms=4)
    ax.axvline(0, color='gray', ls=':'); ax.set_title('arc R=270 mm across the top-left corner line'); ax.set_xlabel('angle from corner line [deg]'); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fdtd_corner.png'), dpi=110)
    for lab in res:
        print(f'{lab:7s} x430:', np.round(res[lab][:len(ys)], 2))
        print(f'{lab:7s} x500:', np.round(res[lab][len(ys):n_line], 2))
        print(f'{lab:7s} arc :', np.round(res[lab][n_line:], 2))
    for lab in ('GO', 'GO+UTD'):
        e = res[lab] - res['FDTD']
        print(f'{lab}: RMS dev vs FDTD  lines {np.sqrt(np.mean(e[:n_line]**2)):.3f}  arc {np.sqrt(np.mean(e[n_line:]**2)):.3f}')

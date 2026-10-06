import numpy as np
from numba import jit

# --- LOW LEVEL JIT COMPILED FUNCTIONS (FAST) ---

@jit(nopython=True, cache=True)
def dot_2d(a, b):
    return a[0]*b[0] + a[1]*b[1]

@jit(nopython=True, cache=True)
def norm_2d(a):
    return (a[0]**2 + a[1]**2)**0.5

@jit(nopython=True, cache=True)
def cross_3d(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1],
                     a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0]])

@jit(nopython=True, cache=True)
def cross_2d(a, b):
    return a[0]*b[1] - a[1]*b[0]

# CHANGED: Default tol reduced from 1.e-3 to 1.e-9
@jit(nopython=True, cache=True)
def _seg_seg_intersect_2d_numba(a1, a2, b1, b2, tol=1.e-9):
    rx = a2[0] - a1[0]
    ry = a2[1] - a1[1]
    sx = b2[0] - b1[0]
    sy = b2[1] - b1[1]
    
    r_cross_s = rx * sy - ry * sx
    
    if abs(r_cross_s) < 1e-12:
        return np.full((2,), np.nan) 
        
    q_px = b1[0] - a1[0]
    q_py = b1[1] - a1[1]
    
    t = (q_px * sy - q_py * sx) / r_cross_s
    u = (q_px * ry - q_py * rx) / r_cross_s
    
    len_r = (rx**2 + ry**2)**0.5
    len_s = (sx**2 + sy**2)**0.5
    
    tol_t = tol / len_r if len_r > 0 else 0
    tol_u = tol / len_s if len_s > 0 else 0
    
    if (-tol_t <= t <= 1.0 + tol_t) and (-tol_u <= u <= 1.0 + tol_u):
        return np.array([a1[0] + t * rx, a1[1] + t * ry])
        
    return np.full((2,), np.nan)

@jit(nopython=True, cache=True)
def _circunf_seg_intersect_2d_numba(circle_center, circle_radius, pt1, pt2, full_line=False, tangent_tol=1e-9):
    cx, cy = circle_center[0], circle_center[1]
    p1x, p1y = pt1[0], pt1[1]
    p2x, p2y = pt2[0], pt2[1]
    
    x1, y1 = p1x - cx, p1y - cy
    x2, y2 = p2x - cx, p2y - cy
    
    dx = x2 - x1
    dy = y2 - y1
    dr_sq = dx * dx + dy * dy
    
    if dr_sq < tangent_tol * tangent_tol:
        dist_sq = x1 * x1 + y1 * y1
        r_sq = circle_radius * circle_radius
        if abs(dist_sq - r_sq) < tangent_tol * tangent_tol:
            res = np.zeros((1, 2))
            res[0, 0] = pt1[0]
            res[0, 1] = pt1[1]
            return res
        else:
            return np.zeros((0, 2))
    
    big_d = x1 * y2 - x2 * y1
    r_sq = circle_radius * circle_radius
    discriminant = r_sq * dr_sq - big_d * big_d

    if discriminant < 0:
        return np.zeros((0, 2))
    
    sqrt_discriminant = discriminant ** 0.5
    sign_dy = -1.0 if dy < 0 else 1.0
    
    ix1 = (big_d * dy + sign_dy * dx * sqrt_discriminant) / dr_sq
    iy1 = (-big_d * dx + abs(dy) * sqrt_discriminant) / dr_sq
    ix2 = (big_d * dy - sign_dy * dx * sqrt_discriminant) / dr_sq
    iy2 = (-big_d * dx - abs(dy) * sqrt_discriminant) / dr_sq
    
    pt_i1 = np.array([cx + ix1, cy + iy1])
    pt_i2 = np.array([cx + ix2, cy + iy2])
    
    temp_res = np.zeros((2, 2))
    valid_count = 0
    
    # Check pt1
    is_valid_1 = True
    if not full_line:
        if abs(dx) > abs(dy):
            t1 = (ix1 - x1) / dx
        else:
            t1 = (iy1 - y1) / dy  
        if not (-1e-5 <= t1 <= 1.0 + 1e-5):
            is_valid_1 = False
            
    if is_valid_1:
        temp_res[valid_count, :] = pt_i1
        valid_count += 1
        
    # Check pt2
    if discriminant > tangent_tol:
        is_valid_2 = True
        if not full_line:
            if abs(dx) > abs(dy):
                t2 = (ix2 - x1) / dx
            else:
                t2 = (iy2 - y1) / dy
            if not (-1e-5 <= t2 <= 1.0 + 1e-5):
                is_valid_2 = False
        
        if is_valid_2:
            temp_res[valid_count, :] = pt_i2
            valid_count += 1
            
    return temp_res[:valid_count]


@jit(nopython=True, cache=True)
def _ellipse_seg_intersect_2d_numba(c, a, b, cos_phi, sin_phi, p1, p2, tol=1.e-9):
    """First crossing of the segment p1->p2 with an ellipse.

    The ellipse has centre ``c``, semi-axes ``a`` (local x) and ``b`` (local y)
    and is rotated by ``phi`` (given through ``cos_phi``/``sin_phi``).

    Returns a length-5 array ``[px, py, nx, ny, inside]``: hit point, outward
    unit normal at the hit and ``inside == 1.0`` when ``p1`` lies inside the
    ellipse.  All-NaN when the segment does not cross the ellipse.  ``tol`` is a
    distance tolerance [mm] on the segment parameter, with the same meaning as
    in ``_seg_seg_intersect_2d_numba``: roots closer than ``tol`` behind ``p1``
    are accepted, roots further behind are not.
    """
    # segment in the ellipse frame, scaled to the unit circle
    vx1 = p1[0] - c[0]
    vy1 = p1[1] - c[1]
    vx2 = p2[0] - c[0]
    vy2 = p2[1] - c[1]
    u1 = (cos_phi * vx1 + sin_phi * vy1) / a
    w1 = (-sin_phi * vx1 + cos_phi * vy1) / b
    u2 = (cos_phi * vx2 + sin_phi * vy2) / a
    w2 = (-sin_phi * vx2 + cos_phi * vy2) / b

    du = u2 - u1
    dw = w2 - w1
    A = du * du + dw * dw
    B = 2.0 * (u1 * du + w1 * dw)
    C = u1 * u1 + w1 * w1 - 1.0

    res = np.full((5,), np.nan)
    if A < 1e-30:
        return res
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return res
    sq = disc ** 0.5
    t_lo = (-B - sq) / (2.0 * A)
    t_hi = (-B + sq) / (2.0 * A)

    rx = p2[0] - p1[0]
    ry = p2[1] - p1[1]
    len_r = (rx * rx + ry * ry) ** 0.5
    tol_t = tol / len_r if len_r > 0 else 0.0

    t = np.nan
    if (-tol_t <= t_lo) and (t_lo <= 1.0 + tol_t):
        t = t_lo
    elif (-tol_t <= t_hi) and (t_hi <= 1.0 + tol_t):
        t = t_hi
    if t != t:
        return res

    # hit point (global) and outward normal from the local gradient
    ul = u1 + t * du
    wl = w1 + t * dw
    gx = ul / a
    gy = wl / b
    gn = (gx * gx + gy * gy) ** 0.5
    gx /= gn
    gy /= gn
    res[0] = p1[0] + t * rx
    res[1] = p1[1] + t * ry
    res[2] = cos_phi * gx - sin_phi * gy
    res[3] = sin_phi * gx + cos_phi * gy
    res[4] = 1.0 if C < 0.0 else 0.0
    return res

# --- PYTHON API WRAPPERS ---

def cross(a, b):
    return cross_3d(a, b)

# CHANGED: Default tol reduced from 1.e-3 to 1.e-9
def seg_seg_intersect_2d(a1, a2, b1, b2, tol=1.e-9):
    res = _seg_seg_intersect_2d_numba(
        np.asarray(a1, dtype=np.float64), 
        np.asarray(a2, dtype=np.float64), 
        np.asarray(b1, dtype=np.float64), 
        np.asarray(b2, dtype=np.float64), 
        tol
    )
    if np.isnan(res[0]):
        return None
    return res

def circunf_seg_intersect_2d(circle_center, circle_radius, pt1, pt2, full_line=False, tangent_tol=1e-9):
    res = _circunf_seg_intersect_2d_numba(
        np.asarray(circle_center, dtype=np.float64),
        float(circle_radius),
        np.asarray(pt1, dtype=np.float64),
        np.asarray(pt2, dtype=np.float64),
        full_line,
        tangent_tol
    )
    
    if res.shape[0] == 0:
        return []
    
    return [row for row in res]


def ellipse_seg_intersect_2d(c, a, b, phi, pt1, pt2, tol=1.e-9):
    """First crossing of segment pt1->pt2 with a rotated ellipse.

    :return: None, or array ``[px, py, nx, ny, inside]`` (see the Numba kernel)
    """
    res = _ellipse_seg_intersect_2d_numba(
        np.asarray(c, dtype=np.float64), float(a), float(b),
        float(np.cos(phi)), float(np.sin(phi)),
        np.asarray(pt1, dtype=np.float64),
        np.asarray(pt2, dtype=np.float64),
        tol
    )
    if np.isnan(res[0]):
        return None
    return res


_ANGLE_TOL = 1.e-9  # [rad] angles this close to pi are snapped to 0 (see wrap_angle_pi)


def wrap_angle_pi(theta):
    """Wrap a propagation angle to [0, pi), the range of the wave-speed tables.

    A direction with a roundoff-level negative y component (e.g. produced by
    reconstructing a direction from a non axis-aligned normal/tangent pair at
    a curved boundary) has ``theta % pi`` equal to pi up to roundoff, which the
    tables treat as a different angle from 0 although it is the same direction.
    Such angles are snapped back to 0.
    """
    theta = theta % np.pi
    if theta >= np.pi - _ANGLE_TOL:
        return 0.
    return theta


@jit(nopython=True, cache=True)
def _point_seg_dist_2d_numba(p, a, b):
    """Distance from point ``p`` to the segment ``a -> b``.

    Returns ``(dist, t)`` with ``t`` in [0, 1] the parameter of the foot of the
    perpendicular clamped to the segment (``a + t*(b - a)``).
    """
    rx = b[0] - a[0]
    ry = b[1] - a[1]
    l2 = rx * rx + ry * ry
    if l2 <= 0.0:
        dx = p[0] - a[0]
        dy = p[1] - a[1]
        return (dx * dx + dy * dy) ** 0.5, 0.0
    t = ((p[0] - a[0]) * rx + (p[1] - a[1]) * ry) / l2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    fx = a[0] + t * rx
    fy = a[1] + t * ry
    dx = p[0] - fx
    dy = p[1] - fy
    return (dx * dx + dy * dy) ** 0.5, t

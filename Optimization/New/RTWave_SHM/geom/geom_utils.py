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
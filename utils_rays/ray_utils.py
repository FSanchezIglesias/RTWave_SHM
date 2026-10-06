from geom.geom_utils import dot_2d, norm_2d, cross_2d
from RayTracing.Ray import Ray, a_tol
import RayTracing.Ray as _ray_module  # for the ``dispersion_follows_direction`` switch
from scipy.fft import rfftfreq
import math
import numpy as np
import logging


def _inherit_kwargs(ray):
    """Diffraction bookkeeping a daughter spawned at ``ray``'s last event inherits."""
    x_here = ray.x[-1]
    law = ray.amp_law
    if law is not None:
        law = (law[0], law[1], law[2], law[3] + x_here)
    return dict(x_src=ray.x_src + x_here, dtheta=ray.dtheta, amp_law=law,
                diff_order=ray.diff_order)


def ray_refl(ray, n, d, intersect, t_int, t,
             ratio, map, bl=0., ratio_mode=1.):
    """ Reflects ray

    :param ray: incident ray
    :param n: object normal
    :param d: object tangent
    :param intersect: Intersection point
    :param t_int: intersection time
    :param t: time calculated when the intersection was detected
    :param ratio: ray power refraction ratio
    :param ratio_mode: ratio between symmetric and antisymmetric
    :param bl: boundary loss
    :param objs: objects in the ray map
    :return: reflected ray, if any
    """
    irays = []

    # Compute ray params at intersection first — needed by caller for refraction
    ray_params_i = ray.calc_ray(t_int, i=-2)
    x_i, trace_i, d_i, f_i, a_i, t_i = ray_params_i

    a_rfl = a_i * ratio * ratio_mode * (1 - bl)
    a_mc = a_i * ratio * (1 - ratio_mode) * (1 - bl)

    # Early-return for fully transparent walls (ratio_rfl=0): skip direction compute + dead trace
    if a_rfl <= a_tol and a_mc <= a_tol:
        ray.set_param(x_i, intersect, d_i, f_i, 0.0, t_int, i=-1)
        return irays, ray_params_i

    # --- Reflection ---
    # Specular reflection
    rfl_dir = - dot_2d(ray.d[-1], n) * n + dot_2d(ray.d[-1], d) * d

    # FIX: Normalize direction and add small offset (nudge)
    rfl_dir_norm = rfl_dir / norm_2d(rfl_dir)
    epsilon = 1e-8
    intersect_safe = intersect + rfl_dir_norm * epsilon

    ray.set_param(x_i,
                  intersect_safe,  # forced to safe offset
                  rfl_dir_norm,    # updated direction
                  f_i,
                  a_rfl,
                  t_int, i=-1)

    # Refresh the dispersion curve for the new (reflected) direction so that
    # the next segment is dispersed with the curve of its own angle
    # (anisotropic media). Disabled -> v1 behaviour (birth-direction curve).
    if _ray_module.dispersion_follows_direction:
        ray._update_phase_coeff()

    # Mode change of reflection:
    if a_mc > a_tol:
        ray_mc = mode_change(ray, a_mc)
        irays.append(ray_mc.__hash__())
        irays.extend(ray_mc.trace(t, map))

    # Propagate original ray to t, once the direction and everything else is modified
    irays.extend(ray.trace(t, map))

    # end function and return refraction or new modes if any
    return irays, ray_params_i


def ray_refr(ray, n, d, intersect, t_int, t, rd, ra, rf,
             ratio, m2, map, bl=0., ratio_mode=1., v2_v1=None):
    """ Refracts ray
     Make sure to execute this always before the reflection!!!

    :param ray: incident ray
    :param n: object normal
    :param d: object tangent
    :param intersect: Intersection point
    :param t_int: intersetion time
    :param t: time calculated when the intersection was detected
    :param rd: ray direction before intersection
    :param ra: ray intensity factor before intersection
    :param rf: ray frequency before intersection
    :param rt: ray time at intersection
    :param ratio: ray power refraction ratio
    :param ratio_mode: ratio between symmetric and antisymmetric
    :param m2: material 2
    :param bl: boundary loss
    :param v2_v1: velocity ratio m2 / incident medium (computed with the
        incident direction by ``_interact``); evaluated here if None
    :return: refracted rays, if any
    """

    # Pre-compute amplitudes to avoid unnecessary work
    a_rfr = ra * ratio * ratio_mode * (1 - bl)
    a_rfr_mc = ra * ratio * (1 - ratio_mode) * (1 - bl)

    # Skip entirely if neither refracted ray would survive
    if a_rfr <= a_tol and a_rfr_mc <= a_tol:
        return []

    # material impedance ratios thing for Snell's law v2/v1
    # TODO: maybe try to fix this for composite
    if v2_v1 is None:
        v2_v1 = m2.v_ray(ray) / \
                ray.medium.v_ray(ray)

    irays = []

    # --- Refraction ---
    # SNELLs law
    # cos_theta_i = dot_2d(ray.d[-1], n)
    sin_theta_i = dot_2d(rd, d)
    # print(ray.d, theta_i*180/np.pi)
    sin_theta_r = v2_v1 * sin_theta_i

    # Refraction exists
    if abs(sin_theta_r) <= 1.:
        # print( ray.d, self.n, np.arcsin(sin_theta_i)*180/np.pi, np.arcsin(sin_theta_r)*180/np.pi)
        rfr_dir = math.cos(math.asin(sin_theta_r)) * n + sin_theta_r * d
        
        # FIX: Normalize direction and add small offset (nudge)
        rfr_dir_norm = rfr_dir / norm_2d(rfr_dir)
        epsilon = 1e-8
        intersect_safe = intersect + rfr_dir_norm * epsilon

        inh = _inherit_kwargs(ray)

        if a_rfr > a_tol:
            # Generate refracted ray
            ray_refr = Ray(intersect_safe, rfr_dir_norm, freq=rf, medium=m2, t=ray.t, t0=t_int, kind=ray.kind,
                           a=a_rfr, parent=ray, _fft_freq=ray.fft_freq, **inh)

            # mode_change must read ray_refr's INITIAL state, before .trace() mutates it
            if a_rfr_mc > a_tol:
                ray_refr_mc = mode_change(ray_refr, a_rfr_mc, parent=ray)

            # Now trace both
            irays.append(ray_refr.__hash__())
            irays.extend(ray_refr.trace(t, map))

            if a_rfr_mc > a_tol:
                irays.append(ray_refr_mc.__hash__())
                irays.extend(ray_refr_mc.trace(t, map))

        elif a_rfr_mc > a_tol:
            # Only the mode-changed refracted ray survives — still need a base ray for mode_change
            ray_refr = Ray(intersect_safe, rfr_dir_norm, freq=rf, medium=m2, t=ray.t, t0=t_int, kind=ray.kind,
                           a=a_rfr, parent=ray, _fft_freq=ray.fft_freq, **inh)
            ray_refr_mc = mode_change(ray_refr, a_rfr_mc, parent=ray)
            irays.append(ray_refr_mc.__hash__())
            irays.extend(ray_refr_mc.trace(t, map))

    # end function and return refraction or new modes if any
    return irays


# def refl_refr(ray, n, d, intersect, t_int, t,
#               v2_v1, ratio, m2, bl=0., ratio_mode=1.):
#     """
#     :param ray: incident ray
#     :param n: object normal
#     :param d: object tangent
#     :param intersect: Intersection point
#     :param t_int: intersetion time
#     :param t: time of intersection
#     :param v2_v1: material impedance ratios thing for Snell's law v2/v1
#     :param ratio: ray power refraction ratio
#     :param ratio_mode: ratio between symmetric and antisymmetric
#     :param m2: material 2
#     :param bl: boundary loss
#     :param objs: objects in the ray map
#     :return: reflected ray, if any
#     """
#     irays = []
#
#     # --- Reflection ---
#     # Specular reflection
#     # rfl_dir = - dot_2d(ray.d, self.n)*self.n + dot_2d(ray.d, self.d)*self.d
#     rfl_dir = - dot_2d(ray.d[-1], n) * n + dot_2d(ray.d[-1], d) * d
#
#     # Replace parameters for intersection point of incident ray
#     x_i, trace_i, d_i, f_i, a_i, t_i = ray.calc_ray(t_int, i=-2)
#
#     ray.set_param(x_i,
#                   intersect,  # forced
#                   rfl_dir / norm_2d(rfl_dir),  # updated direction
#                   f_i,
#                   a_i * ratio * ratio_mode * (1 - bl),
#                   t_int, i=-1)
#
#     # Mode change of reflection:
#     ray_mc = mode_change(ray, ray.a[-1] * ratio * (1 - ratio_mode) * (1 - bl))
#     irays.append(ray_mc)
#     irays.extend(ray_mc.trace(t))
#
#     # --- Refraction ---
#     # SNELLs law
#     # cos_theta_i = dot_2d(ray.d[-1], n)
#     sin_theta_i = dot_2d(ray.d[-1], d)
#     # print(ray.d, theta_i*180/np.pi)
#     sin_theta_r = v2_v1 * sin_theta_i
#
#     # Refraction exists
#     if abs(sin_theta_r) <= 1.:
#         # print( ray.d, self.n, np.arcsin(sin_theta_i)*180/np.pi, np.arcsin(sin_theta_r)*180/np.pi)
#         rfr_dir = math.cos(math.asin(sin_theta_r)) * n + sin_theta_r * d
#
#         # Copy ray color
#         c = ray.color.copy()
#
#         # Generate refracted ray
#         ray_refr = Ray(intersect, rfr_dir, freq=ray.freq[-1].copy(), medium=m2, t=ray.t, t0=t_int,
#                        color=c, a=ray.a[-1] * (1-ratio)*ratio_mode*(1-bl), parent=ray)
#         ray_refr_mc = mode_change(ray_refr, ray.a[-1] * (1-ratio)*(1-ratio_mode)*(1-bl), parent=ray)
#
#         # Propagate rays to t
#         # this is a trace method so new rays could be generated here and must be captured
#         irays.append(ray_refr)
#         irays.append(ray_refr_mc)
#         irays.extend(ray_refr.trace(t))
#         irays.extend(ray_refr_mc.trace(t))
#
#     # Propagate original ray to t, once the direction and everything else is modified
#     irays.extend(ray.trace(t))
#
#     # end function and return refraction or new modes if any
#     return irays


def mode_change(ray, a_new, parent=None):
    """
    Generates a ray with a different mode shape
    :param ray: original ray
    :param a_new: energy / amplitude for the new ray
    :param parent: ray daddy
    :return: new ray with a different mode
    """

    parent = ray if parent is None else parent

    # c[:3] = 1.-c[:3]    # invert color
    # TODO: implement ray.copy() method
    mc_ray = Ray(ray.trace_points[-1].copy(), ray.d[-1].copy(),
                 freq=ray.freq[-1].copy(),
                 medium=ray.medium, t=ray.t, t0=ray.int_times[-1],
                 kind='A0' if ray.kind == 'S0' else 'S0',
                 a=a_new, parent=parent, _fft_freq=ray.fft_freq,
                 **_inherit_kwargs(ray))

    return mc_ray


_DIFF_SECTOR_MARGIN = 1.e-4   # [rad] keep fan rays off the faces of the vertex
_DIFF_PROBE_EPS = 1.e-4       # [mm] offset of the probe rays from the vertex
_DIFF_PROBE_AMIN = 1.e-3      # relative amplitude below which a probe branch is dropped
_DIFF_PROBE_DEPTH = 6         # max number of wall interactions of a probe
_DIFF_PROBE_LFAR = 1.e5       # [mm] probe trace length


def _probe_families(vertex, med, p, d, a, F, fft_freq, kind, fi, depth, out):
    """Trace a probe ray through the walls of ``vertex``'s obstacle.

    Geometrical-optics only (same amplitude rules as ``ray_refl``/``ray_refr``,
    no mode conversion), recursive over reflections and refractions.  The
    spectrum ``F`` is dispersed along every leg with the medium's phase
    coefficient (``medium.phase_coeff``), so families that travelled through
    a slower/thicker region carry their delay and chirp.  Every branch that
    leaves the obstacle is appended to ``out`` as
    ``(medium, direction, amplitude, escape point, spectrum)``.
    """
    p1 = p + d * _DIFF_PROBE_LFAR
    best = None
    for w in vertex.walls:
        h = w.hit(p, p1)
        if h is not None and (best is None or h[0] < best[0][0]):
            best = (h, w)
    if best is None or depth == 0:
        out.append((med, d, a, p, F))
        return
    from RayTracing.Ray import total_internal_reflection
    (s_hit, q, n, dt), w = best
    F_q = np.exp(med.phase_coeff(kind, fft_freq, d) * s_hit) * F
    R = w.ratio_rfl * w.ratio_mode * (1. - w.bl)
    T = (1. - w.ratio_rfl) * w.ratio_mode * (1. - w.bl)
    d_t = None
    if len(w.mediums) == 2:
        m2 = w.mediums[1] if w.mediums[0] is med else w.mediums[0]
        v2_v1 = m2.v_dir(kind, fi, d) / med.v_dir(kind, fi, d)
        sin_t = v2_v1 * dot_2d(d, dt)
        if abs(sin_t) <= 1.:
            d_t = math.cos(math.asin(sin_t)) * n + sin_t * dt
            d_t = d_t / norm_2d(d_t)
        elif total_internal_reflection:
            R = R + T      # same rule as _interact: the transmitted share reflects
    d_r = d - 2. * dot_2d(d, n) * n
    if a * R > _DIFF_PROBE_AMIN:
        _probe_families(vertex, med, q + 1.e-8 * d_r, d_r, a * R, F_q, fft_freq, kind, fi,
                        depth - 1, out)
    if d_t is not None and a * T > _DIFF_PROBE_AMIN:
        _probe_families(vertex, m2, q + 1.e-8 * d_t, d_t, a * T, F_q, fft_freq, kind, fi,
                        depth - 1, out)


def _fan_angles(s0, phi0, max_angle, s_max):
    """Angles ``(dphi, spacing)`` of a diffracted fan around its boundary.

    Rays are spaced ``s0`` within ``phi0`` of the boundary, where the Fresnel
    transition varies fastest, and the spacing then grows linearly with the
    angle (the far diffracted field only decays like 1/angle) up to ``s_max``
    (so that a sensor still sees several fan rays), out to ``max_angle`` on
    both sides.
    """
    out = []
    e = 0.
    while e < max_angle:
        s = min(s_max, s0 * max(1., e / phi0))
        c = e + 0.5 * s
        out.append((c, s))
        out.append((-c, s))
        e += s
    return out


def _point_in_walls(q, walls):
    """Even-odd test of ``q`` against the closed polygon formed by ``walls``."""
    inside = False
    for w in walls:
        a1, a2 = w._a1, w._a2
        if (a1[1] > q[1]) != (a2[1] > q[1]):
            x_int = a1[0] + (q[1] - a1[1]) / (a2[1] - a1[1]) * (a2[0] - a1[0])
            if x_int > q[0]:
                inside = not inside
    return inside


def _inner_medium(vertex):
    """The medium enclosed by the obstacle walls of ``vertex`` (all of its
    objects are walls of the obstacle), or None for an open obstacle."""
    walls = set(id(w) for w in vertex.walls)
    for med in vertex.mediums:
        if med.objs and all(id(o) in walls for o in med.objs):
            # closed only if every wall endpoint is shared by two walls
            ends = []
            for w in vertex.walls:
                ends.extend([w._a1, w._a2])
            for e in ends:
                if sum(norm_2d(e - f) < 1.e-6 for f in ends) != 2:
                    return None
            return med
    return None


def ray_diff(ray, vertex, x_foot, t, map):
    """Launch the diffracted ray fans of ``vertex`` excited by ``ray``.

    ``ray`` is the representative of its family at the vertex (the one whose
    trace passes within half a ray spacing of it, see ``Ray.trace``).  The
    geometrical-optics (GO) field around the vertex is made of ray families
    (the incident one, the ones reflected and transmitted by the faces, the
    ones that entered next to the corner and left through the adjacent
    face, ...) whose boundaries all pass through the vertex; the GO field
    jumps across each boundary and edge diffraction is what keeps the
    physical field continuous there.

    The families are found numerically: two probe rays parallel to ``ray``
    at +/- ``_DIFF_PROBE_EPS`` from the vertex are traced through the walls
    of the obstacle (``_probe_families``), carrying the spectrum of ``ray``
    dispersed along their legs; every outgoing family (medium, direction)
    whose spectrum ``a * F`` differs between the two probes is a shadow
    boundary, with the complex difference spectrum ``dF`` as its jump (a
    family that crossed a slower region is delayed and chirped, so the jump
    between it and the incident wave is not just an amplitude ratio).  A
    transparent wall therefore produces no fan at all.

    For each boundary a fan of rays is launched from the vertex around the
    boundary direction, in the family's medium and restricted to the sector
    that medium occupies at the vertex: spectrum ``dF/2`` (negative on the
    side of the boundary where the GO field is larger, positive on the
    other side, so that GO + fan is continuous), amplitude ``a_i`` scaled by
    the fan/incident angular spacing ratio, times the UTD Fresnel
    transition and the spreading law of ``Ray.amp_factor``.  Fans keep the incident mode (no mode conversion at the
    edge), are spaced ``diff_params['dtheta_factor']`` times the incident
    family and span ``diff_params['max_angle']`` on both sides of the
    boundary.

    :param x_foot: path position of ``ray`` at the vertex
    :return: hashes of the rays spawned
    """
    from RayTracing.Ray import diff_params
    irays = []
    seg = len(ray.x) - 2  # the trace has already appended the segment end
    x_i, trace_i, d_i, f_i, a_i, t_i = ray.calc_ray(i=seg, x=x_foot)
    if not ray.alive:
        return irays

    V = vertex.p
    m = ray.medium
    fi = ray._dom_freq
    p0 = ray.trace_points[seg]

    # --- families on both sides of the vertex --------------------------
    n_ccw = np.array([-d_i[1], d_i[0]])
    off_line = cross_2d(d_i, p0 - V)          # offset of the ray's line from V
    mid = 0.5 * (p0 + trace_i)                # a point of the trace inside m
    x_mid = 0.5 * (ray.x[seg] + x_i)
    f_mid = ray.calc_ray(i=seg, x=x_mid)[3]   # probe start spectrum
    fft_freq = ray.fft_freq
    idom = ray._dom_freq_idx
    others = [mm for mm in vertex.mediums if mm is not m]
    if len(others) > 1:
        logging.warning('%r: more than two mediums, diffraction skipped', vertex)
        return irays
    m_in = _inner_medium(vertex) if others else None
    fam = {}
    for sgn in (1., -1.):
        q = mid + (sgn * _DIFF_PROBE_EPS - off_line) * n_ccw
        # medium of the probe start: a ray passing a convex corner from inside
        # the obstacle has its outer probe outside it (and vice versa)
        med0 = m
        if m_in is not None:
            med0 = m_in if _point_in_walls(q, vertex.walls) else                 (others[0] if m is m_in else m)
        out = []
        _probe_families(vertex, med0, q, d_i, 1., f_mid, fft_freq, ray.kind, fi,
                        _DIFF_PROBE_DEPTH, out)
        for med, d, a, pe, F in out:
            # reference the spectrum to the vertex: propagate along the escape
            # direction up to the foot of the vertex on the escape line, so
            # that all families (and both probes) share the same reference
            F = np.exp(med.phase_coeff(ray.kind, fft_freq, d) * dot_2d(V - pe, d)) * F
            key = (id(med), round(math.atan2(d[1], d[0]), 9))
            entry = fam.setdefault(key, [med, d, None, None, None, None])
            if sgn > 0:
                entry[2] = a * F
                entry[4] = pe
            else:
                entry[3] = a * F
                entry[5] = pe

    # --- sectors of the media at the vertex ------------------------------
    # (``mid`` is strictly inside m; the foot point may lie on a face)
    _, _, _, a_start, extent = vertex.sector(mid - V)
    sectors = {id(m): (a_start, extent)}
    if others:
        sectors[id(others[0])] = (a_start + extent, 2. * np.pi - extent)
    if vertex.single_medium and extent < np.pi + 1.e-9:
        return irays  # closed-boundary corner of a single medium: GO exact

    dth_fan = diff_params['dtheta_factor'] * ray.dtheta
    X = ray.x_src + x_i
    two_pi = 2. * np.pi
    angles = _fan_angles(dth_fan, diff_params['uniform_angle'], diff_params['max_angle'],
                         max(dth_fan, diff_params['max_spacing']))

    beam = getattr(map, 'init_beam', None)
    a_min = diff_params['min_amp']
    if beam is not None:
        a_min = max(a_min, diff_params['min_rel_amp'] * beam.a0)
    f_ref = abs(f_mid[idom])
    for key, (med, u_b, F_p, F_m, pe_p, pe_m) in fam.items():
        if F_p is None:
            F_p = np.zeros_like(f_mid)
        if F_m is None:
            F_m = np.zeros_like(f_mid)
        # complex jump spectrum, from the side where the family is stronger
        # (the fan is -dF/2 on that side and +dF/2 on the other)
        p_high = abs(F_p[idom]) > abs(F_m[idom])
        dF = F_p - F_m if p_high else F_m - F_p
        J = abs(dF[idom]) / f_ref          # jump at the dominant frequency
        if J < diff_params['min_jump'] or 0.5 * J * a_i <= a_min:
            continue
        if id(med) not in sectors:
            continue
        a0, ext = sectors[id(med)]
        # Side of the boundary line where the GO field is larger (the family
        # exists): the family occupies the directions from its boundary, on
        # that side, up to the face that bounds the medium's sector; beyond
        # the face (e.g. behind a screen) it is absent although those
        # directions are still on the same side of the boundary line.
        pe_high = pe_p if p_high else pe_m
        f_fan = 0.5 * dF
        high = np.sign(cross_2d(u_b, pe_high - V))
        if high == 0.:
            continue
        phi_b = math.atan2(u_b[1], u_b[0])
        rel_b = (phi_b - a0) % two_pi
        if rel_b > ext:  # boundary outside the sector: clamp to the nearer end
            rel_b = ext if rel_b - ext < two_pi - rel_b else 0.
        tag = '%d_%.6f' % (id(med) % 100000, phi_b)
        for j, (dphi, s_k) in enumerate(angles):
            phi = phi_b + dphi
            rel = (phi - a0) % two_pi
            if rel < _DIFF_SECTOR_MARGIN or rel > ext - _DIFF_SECTOR_MARGIN:
                continue  # outside the medium's sector at the vertex
            u = np.array([math.cos(phi), math.sin(phi)])
            present = (rel > rel_b) if high > 0. else (rel < rel_b)
            sgn = -1. if present else 1.
            # amplitude proportional to the angular coverage of the ray so that
            # the fan's field (amplitude x ray density) is independent of the
            # non-uniform spacing
            a_fan = a_i * s_k / ray.dtheta
            rd = Ray(V + 1.e-8 * u, u, freq=sgn * f_fan, medium=med, t=ray.t, t0=t_i,
                     kind=ray.kind, a=a_fan, parent=ray, _fft_freq=ray.fft_freq,
                     x_src=0., dtheta=s_k, diff_order=ray.diff_order + 1,
                     _hash_extra=(tag, j))
            k = two_pi * fi / med.v_ray(rd)
            rd.amp_law = (X, dphi, k, 0.)
            irays.append(rd.__hash__())
            irays.extend(rd.trace(t, map))

    return irays


def split_ray(ray, t_ind,
              xmin, xmax, ngridx,
              ymin, ymax, ngridy,
#              err_val=0.001  # 0.1 %
              ):
    """ Computes the ray values on a grid

    :param ray: Ray object
    :param t_ind: time index to compute, int
    :param xmin: x min
    :param xmax: x max
    :param ngridx: number of elements on x axis
    :param ymin: y min
    :param ymax: y max
    :param ngridy: number of elements on y axis
    :param err_val: error value to ignore
    :return:
    """
    z_ray = np.zeros([ngridx, ngridy])
    zi_ray = np.zeros([ngridx, ngridy])

    for i, tr in enumerate(ray.trace_points):

        zi = int((tr[0] - xmin) / (xmax - xmin) * ngridx)
        zk = int((tr[1] - ymin) / (ymax - ymin) * ngridy)
        # zi = np.argmin(np.abs(x_axis-tr[0]))
        # zk = np.argmin(np.abs(y_axis-tr[1]))

        s = ray.signal_at_i(i)  # includes amp_factor for diffracted rays
        if (zi < ngridx) and (zk < ngridy):
            z_ray[zi, zk] += s[t_ind]  # if np.abs(s[t_ind]) > max(
            #    np.abs(s)) * err_val else 0.  # r.a[i]*irfft(r.freq[i], n=len(r.t))[t_ind]
            zi_ray[zi, zk] += 1.

    z_ray[zi_ray > 0] = z_ray[zi_ray > 0] / zi_ray[zi_ray > 0]

    return z_ray, zi_ray


def save_ray(ray, h5file, ray_group='rays'):
    """  Stores ray on hdf5 file.

    :param ray: Ray object
    :param h5file: hdf5 file. Must be opened
    :param ray_group: hdf5 group, default: 'rays'
    :return: None
    """

    # Matrix stuff
    # ray.trace_points  # vector nx2
    # ray.d  # vector nx2
    # ray.freq  # vector nxm

    mat = np.column_stack([ray.a, ray.x, ray.int_times])
    try:
        mat = np.concatenate([mat, ray.trace_points, ray.d, ray.freq], axis=1)
    except:
        logging.error('Unable to save Ray: {: #X}'.format(ray.__hash__()))
        return None

    try:
        dset = h5file.create_dataset(ray_group + '/' + str(ray.__hash__()), data=mat, maxshape=(None, mat.shape[1]))
        # Attributes
        # dset.attrs['t'] = ray.t
        dset.attrs['medium'] = ray.medium.__hash__()
        dset.attrs['kind'] = ray.kind
        dset.attrs['parent'] = ray.parent  # .__hash__()
        # dset.attrs['fftf'] = ray.fft_freq
        dset.attrs['alive'] = ray.alive
        dset.attrs['x_src'] = ray.x_src
        dset.attrs['dtheta'] = ray.dtheta
        dset.attrs['diff_order'] = ray.diff_order
        if ray.amp_law is not None:
            dset.attrs['amp_law'] = np.asarray(ray.amp_law, dtype=float)

    except ValueError:
        # Grows the dataset
        # TODO: REVIEW ALL THIS SHIT
        dset = h5file[ray_group + '/' + str(ray.__hash__())]
        dset.resize(mat.shape[0], axis=0)
        dset[:] = mat


def load_ray(rhash, h5file, rmap, ray_group='rays'):
    """ Loads a ray from an hdf5 file

    :param rhash: Ray identifies hash value
    :param h5file: hdf5 file, must be opened
    :param rmap: ray map
    :param ray_group: hdf5 group, default: 'rays'
    :return: ray object
    """

    dset = h5file.get(ray_group + '/' + str(rhash))
    if dset is not None:
        a, it, tr, d, freq = np.real(dset[:, 0]), np.real(dset[:, 2]), np.real(dset[:, 3:5]),\
            np.real(dset[:, 5:7]), dset[:, 7:]

        medium = rmap.mediums[dset.attrs['medium']]
        kind = dset.attrs['kind']
        t = rmap.init_beam.t

        # Bypass __init__ — populate slots directly from HDF5 data
        ray = Ray.__new__(Ray)
        ray.parent = dset.attrs['parent']
        ray.t = t
        ray.medium = medium
        ray.kind = kind
        ray.alive = dset.attrs['alive']
        ray.a = list(a)
        ray.x = list(np.real(dset[:, 1]))
        ray.int_times = list(it)
        ray.trace_points = list(tr)
        ray.d = list(d)
        ray.freq = list(freq)

        # fft_freq: share from beam if available, else compute
        nfft = freq.shape[1]
        if rmap.init_beam is not None and len(rmap.init_beam.fft_freq) == nfft:
            ray.fft_freq = rmap.init_beam.fft_freq
        else:
            ray.fft_freq = rfftfreq(len(t), d=(t[-1] - t[0]) / len(t))[:nfft]

        # Dispersion curve / phase coefficient: same convention as the live
        # object (last direction when dispersion follows direction, else birth).
        d_ref = ray.d[-1] if _ray_module.dispersion_follows_direction else ray.d[0]
        ray.fft_speed, ray._phase_coeff = ray._dispersion_for_direction(d_ref)

        # Restore cached hash
        ray._hash = rhash

        # Diffraction bookkeeping (absent in files written before it existed)
        ray.x_src = float(dset.attrs.get('x_src', 0.))
        ray.dtheta = float(dset.attrs.get('dtheta', 0.))
        ray.diff_order = int(dset.attrs.get('diff_order', 0))
        law = dset.attrs.get('amp_law', None)
        ray.amp_law = None if law is None else tuple(float(v) for v in law)
        ray._vseen = None

        # Dominant frequency — invariant (fshift only rotates phases)
        ray._dom_freq_idx = np.argmax(np.abs(freq[0]))
        ray._dom_freq = ray.fft_freq[ray._dom_freq_idx]

    else:
        logging.debug('Error loading ray: {: #X}'.format(rhash))
        # Dead ray — skip all expensive computation
        ray = Ray.__new__(Ray)
        ray.parent = 0
        ray.t = rmap.init_beam.t
        ray.medium = next(iter(rmap.mediums.values()))
        ray.kind = 'S0'
        ray.alive = False
        ray.a = [0.]
        ray.x = [0.]
        ray.int_times = [0.]
        ray.trace_points = [np.array([0., 0.])]
        ray.d = [np.array([1., 0.])]
        ray.freq = [np.zeros(1)]
        ray.fft_freq = np.zeros(1)
        ray.fft_speed = np.zeros(1)
        ray._hash = rhash
        ray._dom_freq_idx = 0
        ray._dom_freq = 0.
        ray._phase_coeff = np.zeros(1, dtype=np.complex128)
        ray.x_src = 0.
        ray.dtheta = 0.
        ray.diff_order = 0
        ray.amp_law = None
        ray._vseen = None

    return ray

from geom.geom_utils import dot_2d, norm_2d
from RayTracing.Ray import Ray, a_tol
import RayTracing.Ray as _ray_module  # for the ``dispersion_follows_direction`` switch
from scipy.fft import rfftfreq
import math
import numpy as np
import logging


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
             ratio, m2, map, bl=0., ratio_mode=1.):
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
    :param objs: objects in the ray map
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

        if a_rfr > a_tol:
            # Generate refracted ray
            ray_refr = Ray(intersect_safe, rfr_dir_norm, freq=rf, medium=m2, t=ray.t, t0=t_int, kind=ray.kind,
                           a=a_rfr, parent=ray, _fft_freq=ray.fft_freq)

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
                           a=a_rfr, parent=ray, _fft_freq=ray.fft_freq)
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
                 a=a_new, parent=parent, _fft_freq=ray.fft_freq)

    return mc_ray


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

        s = ray.signal_at_i(i)
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

    return ray
from utils_rays.geom_utils import dot_2d, norm_2d
from RayTracing.Ray import Ray
import math
import numpy as np


def ray_refl(ray, n, d, intersect, t_int, t,
             ratio, h5file, bl=0., ratio_mode=1.):
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

    # --- Reflection ---
    # Specular reflection
    # rfl_dir = - dot_2d(ray.d, self.n)*self.n + dot_2d(ray.d, self.d)*self.d
    rfl_dir = - dot_2d(ray.d[-1], n) * n + dot_2d(ray.d[-1], d) * d

    # Replace parameters for intersection point of incident ray
    ray_params_i = ray.calc_ray(t_int, i=-2)
    x_i, trace_i, d_i, f_i, a_i, t_i = ray_params_i

    ray.set_param(x_i,
                  intersect,  # forced
                  rfl_dir / norm_2d(rfl_dir),  # updated direction
                  f_i,
                  a_i * ratio * ratio_mode * (1 - bl),
                  t_int, i=-1)

    # Mode change of reflection:
    ray_mc = mode_change(ray, a_i * ratio * (1 - ratio_mode) * (1 - bl))
    irays.append(ray_mc.__hash__())
    irays.extend(ray_mc.trace(t, h5file))

    # Propagate original ray to t, once the direction and everything else is modified
    irays.extend(ray.trace(t, h5file))

    # end function and return refraction or new modes if any
    return irays, ray_params_i


def ray_refr(ray, n, d, intersect, t_int, t, rd, ra, rf,
             ratio, m2, h5file, bl=0., ratio_mode=1.):
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

        # Generate refracted ray
        ray_refr = Ray(intersect, rfr_dir, freq=rf, medium=m2, t=ray.t, t0=t_int, kind=ray.kind,
                       a=ra * ratio*ratio_mode*(1-bl), parent=ray)
        ray_refr_mc = mode_change(ray_refr, ra * ratio*(1-ratio_mode)*(1-bl), parent=ray)

        # Propagate rays to t
        # this is a trace method so new rays could be generated here and must be captured
        irays.append(ray_refr.__hash__())
        irays.extend(ray_refr.trace(t, h5file))
        irays.append(ray_refr_mc.__hash__())
        irays.extend(ray_refr_mc.trace(t, h5file))

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
                 a=a_new, parent=parent)

    return mc_ray


def split_ray(ray, t_ind,
              xmin, xmax, ngridx,
              ymin, ymax, ngridy,
              err_val=0.001  # 0.1 %
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
            z_ray[zi, zk] += s[t_ind] if np.abs(s[t_ind]) > max(
                np.abs(s)) * err_val else 0.  # r.a[i]*irfft(r.freq[i], n=len(r.t))[t_ind]
            zi_ray[zi, zk] += 1

    z_ray[zi_ray > 0] = z_ray[zi_ray > 0] / zi_ray[zi_ray > 0]

    return z_ray


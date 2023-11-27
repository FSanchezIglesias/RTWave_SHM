import numpy as np
import math
from utils_rays.geom_utils import seg_seg_intersect_2d, norm_2d, cross_2d
from utils_rays.ray_utils import ray_refl, ray_refr
import logging

# class Plane:
#     def __init__(self, point, normal, color):
#         self.n = normal
#         self.p = point
#         self.col = color
# 
#     def intersection(self, l):
#         d = l.d.dot(self.n)
#         if d == 0:
#             return Intersection( vector(0,0,0), -1, vector(0,0,0), self)
#         else:
#             d = (self.p - l.o).dot(self.n) / d
#             return Intersection(l.o+l.d*d, d, self.n, self)


class medium:
    def __init__(self, ws, th, xi=0.,
                 B=np.array([[1, 0, 0], [0, 1, 0]]), P=np.zeros(3),
                 dispersive=True, theta=0.
                 ):
        """
        Medium definition

        :param ws: wave speed function
        :param th: thickness
        :param xi: damping parameter
        :param B: rotation matrix for plots
        :param P: rotation center for plots
        :param dispersive: If true calculate dispersion
        :para theta: Material angle orientation [rad]
        """

        self.ws = ws
        self.th = th  # mm
        self.xi = xi

        # List of objects contained in the medium
        self.objs = []

        # random value based on medium thickness and xi,
        # because I don't want to implement hashing on the wave speed function
        # So this really doesn't make much sense, but I just want all my mediums to be different....
        # I guess it makes the comparison a bit faster when checking? who knows...
        self.__hash = hash(np.random.rand()*self.xi*self.th)

        self.B = B
        self.P = P

        if dispersive:
            self.fshift = self.fshift_dispersion
        else:
            self.fshift = self.fshift_nd

        self.theta=theta

    def add_objs(self, objs):

        for i, obj in enumerate(objs):
            self.objs.append(obj)
            if hasattr(obj, 'add_medium'):
                obj.add_medium(self)

    def v_ray(self, ray, i=-1, fi=None):
        # if theta is None:
        #     theta = math.atan(ray.d[i][1] / ray.d[i][0])

        if fi is None:
            fi = ray.fft_freq[np.argmax(np.abs(ray.freq[i]))]
        f_d = fi/1.E+6 * self.th
        theta = np.arctan2(ray.d[i][1], ray.d[i][0]) + self.theta
        theta = theta % np.pi  # angles defined between [0, pi)

        # logging.debug('Velocity for ray freq: {:.3e}'.format(f))
        # try:
        return getattr(self.ws, ray.kind)((f_d, theta)) * 1.E3
        # except ValueError as e:
        #     logging.error('Ray freq: {:.3e}'.format(f))
        #     raise e
    
    # def tl(self, ray):
    #     # the ray has already advanced
    #     t = ray.int_times[-1] - ray.int_times[-2]
    #     
    #     a_damp = ray.a[-1] * np.exp(-2*np.pi*ray.freq[-1]*self.xi*t)
    #     return a_damp
    
    def tl(self, ray, i, t):
        # the ray has already advanced
        # Rays only stay on a single medium
        f = ray.fft_freq[np.argmax(np.abs(ray.freq[i]))]
        a_damp = ray.a[i] * np.exp(-2*np.pi*f*self.xi*t)
        return a_damp

    def fshift_dispersion(self, f, x, ray, i=-1):
        """ Dispersion estimation based on the FFT shift property

        :param f: X components of the fourier transform
        :param x: Distance
        :returns f_d: X components of the signal disperse
        """

        t_d = x/ray.fft_speed
#        t_d = np.array([x/self.v_ray(ray, fi=fi_freq, i=i)
#                        for fi_freq in ray.fft_freq])
#         t_d = np.array([x/(getattr(self.ws, ray.kind)((fi_freq/1.E+6 * self.th, theta)) * 1.E3)
#                         for fi_freq in ray.fft_freq])
        f_d = np.exp((0. - 1j) * 2 * np.pi * ray.fft_freq * t_d) * f
        return f_d

    def fshift_nd(self, f, x, ray, i=-1):
        """ fft shift assuming average/non-dispersive velocity. """

        v = self.v_ray(ray, i)
        f_d = np.exp((0. - 1j) * 2 * np.pi * ray.fft_freq * x/v) * f
        return f_d

    def __hash__(self):
        return self.__hash


class Segment:
    def __init__(self, a1, a2, boundary_losses=0.2,
                 color='black',
                 B=np.array([[1,0,0], [0,1,0]]), P=np.zeros(3),
                 ratio_rfl=0.8, ratio_mode=0.9):

        self.a1 = a1
        self.a2 = a2
        
        self.d = (a2-a1)/norm_2d(a2-a1)
        self.n = np.array([self.d[1], -self.d[0]])

        self.length = norm_2d(self.a2-self.a1)
        
        self.color = color

        # No more of this
        # self.npos = None
        # self.nneg = None

        # List of mediums that contain the segment
        self.mediums = []

        # energy factor for each of the mediums
        self.ratio_rfl = ratio_rfl
        self.ratio_mode = ratio_mode
        self.bl = boundary_losses

        self.B = B
        self.P = P

    def add_medium(self, medium):
        """ Adds medium to list
        """
        self.mediums.append(medium)

    def intersect(self, ray, t, map):
        """
        Intersects last trace of ray with self

        :param ray: ray that intersects
        :param t: time of analysis
        :return: reflected/refracted new rays
        """

        # get intersections and kill/spawn rays at intersections
        intersect = seg_seg_intersect_2d(ray.trace_points[-2], ray.trace_points[-1], self.a1, self.a2)

        if intersect is not None:
            irays = []
            # Find out where is the ray coming from:
            # theta_n = math.atan(self.n[1]/self.n[0])  # assumes incident speed ratio at normal direction

            # for i, m in enumerate(self.mediums):
            #     if m == ray.medium:
            #         ratio = self.ratio[i]/sum(self.ratio)

            # if ratio is None:
            #     raise TypeError('Impossible intersection found for ray: {} on segment {}\n check medium definition'
            #                     .format(ray, self))

            if cross_2d(self.d, ray.trace_points[-2]-self.a1) > 0:
                n = self.n
                d = self.d

            else:
                # the normal points to the side from where the ray comes,
                # because trigonometry is easier this way
                n = -self.n
                d = -self.d

            # estimate intersc. time
            # d_int = dot_2d(ray.d, i)
            t_int = norm_2d(intersect-ray.trace_points[-2]) / norm_2d(ray.trace_points[-1]-ray.trace_points[-2]) \
                    * (t-ray.int_times[-2]) + ray.int_times[-2]
            # ray parameters at intersection
            # rd, ra, rf, rt = ray.d[-1], ray.a[-1], ray.freq[-1].copy(), ray.t

            # Reflect ray
            # ratio_rfl = self.ratio_rfl if len(self.mediums) > 1 else 1.  # if only one medium all is reflected -> NO!
            irays_rfl, ray_params_i = ray_refl(ray, n, d, intersect, t_int, t,
                                               ratio=self.ratio_rfl, ratio_mode=self.ratio_mode,
                                               bl=self.bl, map=map)
            irays.extend(irays_rfl)

            # Refract ray on all remaining boundaries
            if len(self.mediums) > 1:
                x_i, trace_i, d_i, f_i, a_i, t_i = ray_params_i
                ratio_rfr = (1 - self.ratio_rfl) / (len(self.mediums) - 1)  # self.ratio[i] / sum(self.ratio)
                for i, m2 in enumerate(self.mediums):
                    if not m2 == ray.medium:
                        irays.extend(ray_refr(ray, n, d, intersect, t_int, t, d_i, a_i, f_i,
                                              ratio=ratio_rfr, m2=m2, ratio_mode=self.ratio_mode,
                                              bl=self.bl, map=map))
            # # generate opposite kind echo?
            # if rfr_ray is not None:
            #     irays.append(rfr_ray)

            return irays
        return []

    def plot(self, ax, color='default', marker=None):
        if color == 'default':
            color = self.color
        if color is None:
            return  # do not plot either if color is None or self.color is None

        a1 = self.B.T.dot(self.a1) + self.P
        a2 = self.B.T.dot(self.a2) + self.P

        ax.plot([a1[0], a2[0]], [a1[1], a2[1]], color=color, marker=marker)

    def plot3d(self, ax, color='default', marker=None):
        if color == 'default':
            color = self.color
        if color is None:
            return  # do not plot either if color is None or self.color is None

        a1 = self.B.T.dot(self.a1) + self.P
        a2 = self.B.T.dot(self.a2) + self.P

        ax.plot([a1[0], a2[0]], [a1[1], a2[1]], [a1[2], a2[2]],
                color=color, marker=marker)

    def to_vtk(self):
        """ Returns a string in vtk format

        :return:
        """


class Circunf:
    def __init__(self, c, r, color='black'):
        self.c = c
        self.r = r
               
        self.color = color
        
        self.npos=None
        self.nneg=None
    
    def plot(self, ax, color=None, marker=None):
        if color is None:
            color = self.color

        from matplotlib.pyplot import Circle
        c = Circle(self.c, self.r, color=color, fill=False, zorder=2)
        ax.add_patch(c)

        if marker is not None:
            ax.plot(self.c[0], self.c[1], marker=marker, color=color)


class Polygon:
    def __init__(self, segs, color='black'):
        """
        
        :param segs: list of segments or circles
        
        No checks implemented!!!!
        Segments must be closed, if any of the segments is a circle it has to be completely inside the enclosure
        
        """
        self.segs = segs
        self.color = color
        for seg in self.segs:
            if not isinstance(seg, Circle):
                self._assign_seg_props(seg)
            else:
                seg.npos = self
    
    def centroid(self):
        points = []
        
        for seg in self.segs:
            if not isinstance(seg, Circle):
                points.append(seg.a1)
                points.append(seg.a2)
        
        c = np.array([0,0,0])
        for p in points:
            c+=p
        
        return c / len(points)
        
    def _assign_seg_props(self, seg):
        c = self.centroid()
        
        if cross_2d(seg.a1-c, seg.a2-c) > 0:
            seg.npos = self
        else:
            seg.nneg = self
    
    def plot(self, ax):
        for seg in self.segs:
            seg.plot(ax, color=self.color)


class Circle:
    def __init__(self, circunf, color='black'):
        self.segs = [circunf, ]
        self.color = color
        circunf.nneg = self

    def centroid(self):
        return self.segs[0].c

    def plot(self, ax):
        for seg in self.segs:
            seg.plot(ax, color=self.color)


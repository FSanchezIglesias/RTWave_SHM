from geom.objects_2d import Segment, Circunf
from utils_rays.geom_utils import seg_seg_intersect_2d, circunf_seg_intersect_2d, dot_2d
import numpy as np
from scipy.signal.windows import hamming
import logging


class _StrBoundary(Segment):
    def intersect_sens(self, ray):
        intersect = seg_seg_intersect_2d(ray.trace_points[-2], ray.trace_points[-1], self.a1, self.a2)
        if intersect is None:
            return None

        # t_int = norm_2d(intersect - ray.trace[-2]) /
        # norm_2d(ray.trace[-1] - ray.trace[-2]) * (t - ray.int_times[-2]) \
        #         + ray.int_times[-2]
        x_int = ray.x[-2] + dot_2d(intersect - ray.trace_points[-2], ray.d[-2])

        return [x_int, ]  # return as a list for compatibility


class _CircBoundary(Circunf):
    def intersect_sens(self, ray):

        intersect = circunf_seg_intersect_2d(self.c, self.r, ray.trace_points[-2], ray.trace_points[-1])
        if not intersect:
            return None

        # t_int = norm_2d(intersect - ray.trace[-2]) / norm_2d(ray.trace[-1] - ray.trace[-2]) *
        # (t - ray.int_times[-2]) \
        #         + ray.int_times[-2]
        x_int = []

        # Distance travelled over ray
        for i in intersect:
            x_int.append(ray.x[-2] + dot_2d(i - ray.trace_points[-2], ray.d[-2]))

        return x_int


class Sensor:

    def __init__(self, kind, params, sensitivity=1.,
                 color='red', **kwargs):
        """
        Sensor is for now a rectangle defined by 4 corners
        :param kind: string:
                - 'rect' - Rectangle
                - 'sq' - Square
        :param params: parameters to define the sensor boundary:
                - rectangle: 4 points [[a1, a2], [b1, b2], [c1, c2], [d1, d2]] ordered in a rhs motion
                - square: center and radius [[o1, o2], r]-> to form a square tho...
        """
        self.bounds = []
        self.kind = kind.lower()

        if 'rect' in self.kind:
            self.bounds.append(_StrBoundary(params[0], params[1], color=color))
            self.bounds.append(_StrBoundary(params[1], params[2], color=color))
            self.bounds.append(_StrBoundary(params[2], params[3], color=color))
            self.bounds.append(_StrBoundary(params[3], params[0], color=color))

            self.size = np.average([n.length for n in self.bounds])

        elif 'sq' in self.kind:
            o = np.array(params[0])
            r = params[1]
            p = [o + np.array([-r, -r]), o + np.array([r, -r]),
                 o + np.array([r, r]), o + np.array([-r, r])]

            self.bounds.append(_StrBoundary(p[0], p[1], color=color))
            self.bounds.append(_StrBoundary(p[1], p[2], color=color))
            self.bounds.append(_StrBoundary(p[2], p[3], color=color))
            self.bounds.append(_StrBoundary(p[3], p[0], color=color))

            self.size = np.average([n.length for n in self.bounds])
        elif 'circ' in self.kind:
            o = np.array(params[0])
            r = params[1]
            self.bounds.append(_CircBoundary(o, r, color=color))

            self.size = 2*r

        else:
            raise NotImplementedError('Unknown sensor of kind: {}'.format(kind))

        self.sensitivity = sensitivity/self.size  # hmmmm....

        # # Maybe this should be something else...
        # self.int_size = kwargs.get('int_size', self.size / kwargs.get('int_n', 10))

        self.int_rays = {}
        self.map = None
        self.medium = None

    def intersect(self, ray, t, h5file):
        """ Checks for intersections but rays are not altered
        :param ray: Ray object
        :param t: maintains sig of other intersect methods
        :param h5file: maintains sig of other intersect methods
        :return: empty list
        """

        for b in self.bounds:
            int_points = b.intersect_sens(ray)
            if int_points is not None:
                if ray.__hash__() in self.int_rays.keys():
                    for xi in int_points:
                        self.int_rays[ray.__hash__()].append(xi)
                else:
                    self.int_rays[ray.__hash__()] = int_points

        return []

    def _signal_on_ray(self, ray, xs_ray, d_x, window=None):
        # if len(xs_ray) % 2:
        #     # odd number of cuts, this is bad
        #     # print('error on ray{}'.format(ray.ID))
        #     continue

        xs_ray = set(xs_ray)
        xs_ray = sorted(xs_ray)  # make a set and sort all items

        sig_i = []

        # get all the segments:
        # xs_ray is (should be) always ordered in pairs
        # ( A ray may cut the sensor more than twice )
        for i in range(len(xs_ray) // 2):
            # x_sig = 0.5 * (xs_ray[2*i+1] + xs_ray[2*i])
            xi = np.arange(xs_ray[2 * i], xs_ray[2 * i + 1], d_x) + d_x / 2
            D_ray = np.abs(xs_ray[2 * i] - xs_ray[2 * i + 1])
            
            if window == 'hamming':
                w = hamming(xi.size)
            # elif window == 'double_hamming':
            #     w = hamming(xi.size)
            #     w * /self.size
            elif window == 'hsphere':
                w = hamming(xi.size)
                w *= D_ray/self.size
            else:
                w = np.ones(xi.shape)
            # l_sig = abs(xs_ray[2*i+1] - xs_ray[2*i])
            # int_c = math.ceil(l_sig/d_x)
            # li_sig = l_sig/int_c  # do this again because int
            # for x_int in np.linspace(xs_ray[2*i], xs_ray[2*i+1], int_c+1)[:-1] + li_sig/2:
            for i_w, x in enumerate(xi):
                # sig_i = integral(t, ray.signal_at_x(t, x_int))  # Integrate the signal in time
                try:
                    sig_i.append(ray.signal_at_x(x) * d_x * w[i_w])
                except StopIteration:
                    logging.error('Unable to get signal for ray: {} at x: {:.3f}'.format(hash(ray), x))
                    continue

        return sum(sig_i)

    def signal(self, d_x=0.1, procs=None, window='hsphere'):
        """ Measure signal at sensor
        Considers only rays that cut twice
        :param t: time points
        :param d_x: integration step
        :return: measured signal
        """

        if not self.int_rays:
            return None

        # Time must be equal for all model, its defined on a random ray
        t = self.map.init_beam.t
        signal_mat = np.zeros([len(t), len(self.int_rays)])

        if (procs is None) or (procs == 1):
            for i, [rayh, xs_ray] in enumerate(self.int_rays.items()):

                ray = self.map.get_ray(rayh)

                signal_mat[:, i] = self._signal_on_ray(ray, xs_ray, d_x, window)

        else:
            from multiprocessing import Pool
            p = Pool(procs)

            results = {}
            for i, [ray, xs_ray] in enumerate(self.int_rays.items()):
                args = (ray, xs_ray, d_x, window)
                results[i] = p.apply_async(self._signal_on_ray, args)

            for i, res in results.items():
                signal_mat[:, i] = res.get()

        signal = signal_mat.sum(axis=1)

        return signal * self.sensitivity

    # def
    
    def plot(self, ax, color=None, marker=None):
        for b in self.bounds:
            b.plot(ax, color, marker)

    def origin(self):
        """ Returns center of sensor. Only works for circ, for now...

        :return: array shape 2x1
        """
        if 'circ' in self.kind:
            return self.bounds[0].c
        else:
            raise NotImplementedError('Method not implemented for sensor fo kind {}'.format(self.kind))

    def add_medium(self, medium):
        if self.medium is None:
            self.medium = medium
        else:
            raise TypeError('Unable to redefine medium')

import numpy as np
from scipy.fft import rfft, rfftfreq, irfft
from utils_rays.geom_utils import norm_2d
# from utils_rays.ray_utils import save_ray
from RayTracing.Signal import burst_hann
import logging

a_tol = 1.e-8  # tolerance for living (ABSOLUTE)
ray_color = np.array((0.2, 0.6, 0.2, 0.7))  # default ray color


def alive_ray(ray, i=-1):
    return (ray.a[i] > a_tol) and ray.alive


class Ray:
    __slots__ = ('parent', 't',
                 'medium', 'kind', 'a', 'x',
                 'trace_points', 'd', 'int_times', 'freq',
                 'fft_freq', 'fft_speed', 'alive')

    def __init__(self, origin, direction, freq, medium, t,
                 kind='S0', t0=0., a=1.,
                 **kwargs):
        """

        :param origin: Ray origin, point.
        :param direction: Ray direction, vector.
        :param freq: fft terms.
        :param medium: Medium object from which the ray is propagating.
        :param t: time vector for results.
        :param kind: 'A0' or 'S0'
        :param t0: Spawn time.
        :param color: ray color.
        :param a: ray intensity parameter, default 1.
        :param a_tol: Tolerance for living (ABSOLUTE), float. Default: 1.
        :param norm_c: Normalization value for color plots.
        :param kwargs: Additional keyword arguments.
                - 'parent': Ray
        """

        p = kwargs.get('parent', None)
        if p is not None:
            self.parent = int(p.__hash__())
        else:
            self.parent = 0

        # self.o = origin
        # self.t0 = t0
        self.t = t
        # self.dt = (t[-1]-t[0])/len(t)

        self.medium = medium
        self.kind = kind

        # Ray intensity parameter
        self.a = [a, ]

        self.x = [0., ]
        self.trace_points = [origin, ]
        self.d = [direction / norm_2d(direction), ]

        self.int_times = [t0, ]
        self.freq = [freq, ]
        # self.nfft = len(freq)
        self.fft_freq = rfftfreq(len(t), d=(t[-1]-t[0])/len(t))[:len(freq)]
        self.fft_speed = np.array([self.medium.v_ray(self, fi=fi) for fi in self.fft_freq])

        self.alive = True

    def v_ray(self, i=-1):
        return self.medium.v_ray(self, i)

    def calc_ray(self, t=None, i=-1, x=None):    # t0, x0, f0, a0, t, d):
        """ Calculates the ray propagation between points

        :param t: time to advance
        :param i: index to check, by default is last
        :param x: position on ray
        :return: x, trace_i, d_i, f_i, a_i, t
        """

        t0 = self.int_times[i]
        trace0 = self.trace_points[i]
        x0 = self.x[i]
        f0 = self.freq[i]
        a0 = self.a[i]
        d = self.d[i]

        v = self.v_ray(i)  # only freq / when i=-1 gets previous inc

        if x is None:
            x_i = v * (t - t0)
        elif t is None:
            # time of the signal to reach x
            x_i = x - x0
            t = t0 + x_i/v
        else:
            raise TypeError("Only one argument: 't' or 'x' must be defined")
        # else:
        #    raise TypeError("Missing 1 required keyword argument: 't' or 'x'")

        if t < t0:
            logging.debug('\n'.join(['{:.2e} - ({:.2e}, {:.2e})'.format(w1, w2[0], w2[1])
                                     for w1, w2 in zip(self.int_times, self.trace_points)]))
            if x is not None:
                errormsg = 't value: {:.4e} smaller than last increment {}: {:.4e}, for x: {:.3e}'.format(t, i, t0, x)
            else:
                errormsg = 't value: {:.4e} smaller than last increment {}: {:.4e}'.format(t, i, t0)

            # raise TypeError(errormsg)
            logging.error(errormsg)
            logging.error('Ray {} is now dead, forever'.format(self))
            self.alive = False
            return x0, trace0, d, f0, a0, t

        trace_i = trace0 + d * x_i

        # Implement direction change?
        d_i = d

        # Dispersion and time shift
        f_i = self.medium.fshift(f0, x_i, self, i)

        # Transmission loss
        a_i = self.medium.tl(self, i, t - t0)

        return self.x[i] + x_i, trace_i, d_i, f_i, a_i, t

    def trace(self, t, map):
        """

        :param t: time to advance ray (sorry)
        :return: reflected rays
        """
        # Only trace if you're alive
        if not alive_ray(self):
            return []

        # Advance the ray linearly
        x_i, trace_i, d_i, f_i, a_i, t_i = self.calc_ray(t)

        self.set_param(x_i, trace_i, d_i, f_i, a_i, t_i)

        # Intersect the ray with whatever is in the way
        # BEWARE recursion!!!
        # -- keep track of new rays --
        rfr_rays = []

        for obj in self.medium.objs:
            rfr_rays.extend(obj.intersect(self, t, map))

        map.save_ray(self)  # saves ray
        return rfr_rays

    def retrace(self, length, map):
        """ Calculates additional n points inbetween traces

        # :param n: number of points
        :param length: "approximate" length for retracing
        :return: None
        """

        int_times, x, trace_points, d, freq, a = [], [], [], [], [], []

        for i in range(len(self.int_times)-1):
            ti, tf = self.int_times[i], self.int_times[i+1]
            int_times.append(ti)
            x.append(self.x[i])
            trace_points.append(self.trace_points[i])
            d.append(self.d[i])
            freq.append(self.freq[i])
            a.append(self.a[i])

            n = int((self.x[i+1] - self.x[i]) / length)
            t_inc = (tf-ti)/n
            for ni in range(n):
                tn = ti + ni*t_inc
                xi, trace, di, f, ai, t = self.calc_ray(tn, i)
                int_times.append(t)
                x.append(xi)
                trace_points.append(trace)
                d.append(di)
                freq.append(f)
                a.append(ai)
        # Add last point
        int_times.append(self.int_times[-1])
        x.append(self.x[-1])
        trace_points.append(self.trace_points[-1])
        d.append(self.d[-1])
        freq.append(self.freq[-1])
        a.append(self.a[-1])

        self.int_times = int_times
        self.x = x
        self.trace_points = trace_points
        self.d = d
        self.freq = freq
        self.a = a

        map.save_ray(self)  # saves ray

    def set_param(self, x, trace, d, f, a, t, i=None):
        """ Sets the ray parameters after each iteration
        Kind of safety thing, because it just appends, but just to make sure you don't forget anything

        :param x: X distance
        :param trace: Position
        :param d: Direction vector
        :param f: frequency
        :param a: amplitude
        :param t: time of iteration
        :param i: index, if None, append
        :return: None
        """
        
        if i is None:
            self.int_times.append(t)
            self.x.append(x)
            self.trace_points.append(trace)
            self.d.append(d)
            self.freq.append(f)
            self.a.append(a)
        else:
            # modify ray properties
            self.int_times[i] = t
            self.x[i] = x
            self.trace_points[i] = trace
            self.d[i] = d
            self.freq[i] = f
            self.a[i] = a

    # def end_ray(self, p=None, t=None):
    #     """ kills the ray at a point p and time t
    #       -- DEPRECATED --
    #     :param p: point 2D array
    #     :param t: time of death, float
    #     """
    #     if p is None:
    #         p=self.trace[-1]
    #     if t is None:
    #         t = self.int_times[-1]
    #     self.trace = lambda x: p
    #     # This doesn't check shit, so be careful....
    #     # The last point of the ray is substituted by p
    #     self.trace[-1] = p
    #     self.int_times[-1] = t
    #     self.a[-1] = 0.

    def signal_at_x_f(self, x=0.):
        """ Returns the frequency and intensity signal parameters along a point during the ray path

        :param x: position along ray
        """
        # Index in the ray integration points

        if x in self.x:
            i = self.x.index(x)
            a_i, f_i, t_i = self.a[i], self.freq[i], self.int_times[i]
        else:
            i = next(i for i, v in enumerate(self.x) if v > x) - 1
            x_i, trace_i, d_i, f_i, a_i, t_i = self.calc_ray(i=i, x=x)
        # debug
        # print(i, x_i, x)

        # Estimate ray stuff at x
        # x_i, trace_i, d_i, f_i, a_i = self.calc_ray(self, t0, x0, f0, a0, t, d)
        # arguments to recover signal
        # args = [self.a[i], *self.freq[i]]
        return a_i, f_i, t_i

    def signal_at_x(self, x=0.):
        """ Returns the signal along a point during the ray path

        :param x: position along ray
        """

        a_i, f_i, t_i = self.signal_at_x_f(x)
        s = a_i*irfft(f_i, n=len(self.t))
        # everything before the time in which the ray reaches x must be 0
        # solves weird fft issues
        # tz = np.ones(self.t.shape)
        # t/2 to account for dispersion and still get rid of weird stuff safely
        s[self.t < t_i/2] = 0
        # s *= tz

        return s

    def signal_at_i(self, i=-1):
        """ Returns the signal along a known point i during the ray path

        :param i: point index
        """

        a_i, f_i, t_i = self.a[i], self.freq[i], self.int_times[i]
        s = a_i*irfft(f_i, n=len(self.t))
        # everything before the time in which the ray reaches x must be 0
        # solves weird fft issues
        # tz = np.ones(self.t.shape)
        s[self.t < t_i] = 0
        # s *= tz
        return s

    def plot(self, ax, marker=None, color=None, norm=None, linestyle='-'):

        if color is None:
            color = ray_color.copy()


        B = self.medium.B
        P = self.medium.P
        tp = [B.T.dot(ti) + P for ti in self.trace_points]

        for i in range(len(tp) - 1):

            if alive_ray(self, i):
                try:
                    c = color.copy()  # copy color for each segment
                except AttributeError:
                    c = color

                if norm:
                   c[3] *= self.a[i]/norm

                ax.plot([tp[i][0], tp[i + 1][0]],
                        [tp[i][1], tp[i + 1][1]],
                        color=c, marker=marker, linestyle=linestyle)

    def plot3d(self, ax, marker=None,
               color=np.array([1., 0., 0., 1.]), norm=True):

        B = self.medium.B
        P = self.medium.P
        tp = [B.T.dot(ti) + P for ti in self.trace_points]

        for i in range(len(tp) - 1):

            if alive_ray(self, i):
                if not norm:
                     color[3] *= self.a[i]

                ax.plot([tp[i][0], tp[i + 1][0]],
                        [tp[i][1], tp[i + 1][1]],
                        [tp[i][2], tp[i + 1][2]],
                        color=color, marker=marker)

    def __hash__(self, *args, **kwargs):
        # define ray hash - Kind + direction + origin + birthdate + freq parameters
        # hash of hashes to avoid super large python ints
        return hash(hash(self.kind) + hash(tuple(self.d[0])) + \
                    hash(tuple(self.trace_points[0])) + hash(self.int_times[0]) + hash(tuple(self.freq[0])))

    def __repr__(self):
        return 'Ray {: #X}'.format(self.__hash__())


class Beam:
    def __init__(self, n_rays, params, medium,
                 signal_f=burst_hann, freq=None,
                 power=1., kind='all', b_type='circ', **kwargs):
        """

        :param n_rays: Number of rays
        :param params: Parameters for beam.
                      * if b_type = 'circ': params = [origin, theta_i, theta_f, d]
        :param medium: Medium for the rays
        :param signal_f: Signal function
        :param freq: fft input of ray signal (Normalized)
        :param power: Power on beam
        :param kind: 'S0', 'A0' or 'both'
        :param b_type: Type of beam, only 'circ' is currently supported
        :param kwargs: Other keyword arguments
                     - f: Input signal burst frequency
                     - fd imput signal freq at 1/3
                     - t: time to calc fft of input signal (Recommended analysis time)
                     - nfft: number of terms of fourier transform
        """

        self.rays = []
        self.n_rays = n_rays

        if kind not in ['S0', 'A0']:
            self.a0 = power / (2 * self.n_rays)
        else:
            self.a0 = power / self.n_rays

        if freq is None:
            f = kwargs.get('f', 300.e3)
            fd = kwargs.get('fd', None)
            if fd is None:
                npeaks = kwargs.get('npeaks', 3)
                fd = f / npeaks

            # default is 5 periods with 100 points per period
            self.t = kwargs.get('t', np.arange(0, 1/fd*5, 1/(100*f)))
            self.nfft = kwargs.get('nfft', 50)
            s = signal_f(self.t, self.a0, f, fd)

            self.freq = rfft(s)[:self.nfft]
        else:
            if 't' not in kwargs:
                raise TypeError('A time vector, t, must be defined with freq.')
            self.t = kwargs['t']
            self.nfft = kwargs.get('nfft', 50)

        self.fft_freq = rfftfreq(len(self.t), d=(self.t[-1] - self.t[0]) / len(self.t))[:len(self.freq)]

        if b_type == 'circ':
            # params = [origin, theta_i, theta_f, d]
            self.o = params[0]

            if len(params) == 1:
                theta_i, theta_f = 0, 2 * np.pi
                d = np.array([1, 0])
            else:
                theta_i, theta_f = params[1], params[2]
                if len(params) < 4:
                    d = np.array([1, 0])
                else:
                    d = params[3] / norm_2d(params[3])

            # -- Ray color --  -> NO
            # import matplotlib.pyplot as plt
            # cmap = plt.cm.get_cmap(kwargs.get('cmap_name', 'jet'))
            # color = kwargs.get('color', None)
            # colors = plt.cm.jet(np.linspace(0, 1, self.n_rays))

            theta_d = np.arctan(d[1] / d[0])
            for i in range(self.n_rays):

                theta = (theta_f - theta_i) / self.n_rays * i + theta_d
                d = np.array([np.cos(theta_i + theta), np.sin(theta_i + theta)])

                if kind not in ['S0', 'A0']:
                    ray_a = Ray(origin=self.o, direction=d, freq=self.freq, t=self.t,
                                medium=medium, kind='A0', a=self.a0, nfft=self.nfft)
                    ray_s = Ray(origin=self.o, direction=d, freq=self.freq, t=self.t,
                                medium=medium, kind='S0', a=self.a0, nfft=self.nfft)
                    self.rays.append(ray_a)
                    self.rays.append(ray_s)
                else:
                    ray_i = Ray(origin=self.o, direction=d, freq=self.freq, t=self.t,
                                medium=medium, kind=kind, a=self.a0, nfft=self.nfft)
                    self.rays.append(ray_i)

    def inp_signal(self):
        return irfft(self.freq, n=len(self.t))*self.a0

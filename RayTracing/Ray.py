import numpy as np
from scipy.fft import rfft, rfftfreq, irfft
from geom.geom_utils import norm_2d, wrap_angle_pi, _point_seg_dist_2d_numba
# from utils_rays.ray_utils import save_ray
from RayTracing.Signal import burst_hann
import logging

a_tol = 1.e-8  # tolerance for living (ABSOLUTE)
ray_color = np.array((0.2, 0.6, 0.2, 0.7))  # default ray color

# If True, a ray's dispersion curve (``fft_speed``/``_phase_coeff``) is
# recomputed for the new propagation direction after every reflection, so an
# anisotropic medium disperses each segment with the curve of its own angle.
# If False, the curve of the birth direction is kept for the ray's whole life
# (behaviour of the original v1 solver).
dispersion_follows_direction = True

# Edge (corner) diffraction: the one ray of each family that passes within
# half a ray spacing of a wall vertex launches fans of diffracted rays that
# keep the field continuous across the shadow boundaries of the vertex
# (see utils_rays.ray_utils.ray_diff and utils_rays.utd).  ``diff_params``:
#   dtheta_factor -- angular spacing of a fan (next to its boundary) relative
#                    to the incident family
#   uniform_angle -- [rad] beyond this angle from the boundary the spacing
#                    grows linearly with the angle, up to max_spacing [rad]
#   max_angle     -- half-width [rad] of a fan around its shadow boundary
#   max_order     -- rays of this diffraction order (or higher) do not diffract
#   min_jump      -- shadow boundaries with a smaller GO jump (relative to the
#                    incident amplitude) are ignored
#   min_amp       -- fans whose amplitude at the boundary (half the jump times
#                    the incident amplitude) is below this are not spawned
#   min_rel_amp   -- same threshold relative to the source ray amplitude
#                    (``Beam.a0``): weak families (multiply reflected, ...)
#                    do not diffract
diffraction = True
# Total internal reflection: when a ray cannot refract into the medium behind
# a wall (Snell gives |sin| > 1) the transmitted share is reflected too, so
# the wall reflects (1 - bl) of the amplitude instead of ratio_rfl * (1 - bl)
# and the energy is not silently lost (v1 behaviour, ``False``).
total_internal_reflection = True
diff_params = dict(dtheta_factor=2., uniform_angle=np.deg2rad(3.), max_spacing=np.deg2rad(1.5),
                   max_angle=np.pi / 2., max_order=1, min_jump=0.02, min_amp=10. * a_tol,
                   min_rel_amp=0.05)
_ray_diff = None  # lazily imported utils_rays.ray_utils.ray_diff (circular import)


def alive_ray(ray, i=-1):
    if ray.medium is not None:
        return (ray.a[i] > a_tol) and ray.alive
    else:
        return False


class Ray:
    __slots__ = ('parent', 't',
                 'medium', 'kind', 'a', 'x',
                 'trace_points', 'd', 'int_times', 'freq',
                 'fft_freq', 'fft_speed', 'alive', '_hash',
                 '_dom_freq_idx', '_dom_freq', '_phase_coeff',
                 'x_src', 'dtheta', 'amp_law', 'diff_order', '_vseen')

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
                - 'x_src': path length [mm] from the origin of the ray family
                  (the source or the diffracting vertex) to this ray's birth
                  point; the local ray spacing is ``dtheta * (x_src + x)``
                - 'dtheta': angular spacing [rad] of the ray family (0: the
                  ray never diffracts)
                - 'amp_law': None, or ``(X, dphi, k, x_off)`` for a diffracted
                  ray: the amplitude is multiplied at path position ``x`` by
                  ``amp_factor(x)`` (see there)
                - 'diff_order': diffraction order (0 for geometrical rays)
                - '_hash_extra': extra hashable term to make the hash unique
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
        # float64 arrays so that intersection kernels can be called directly
        self.trace_points = [np.ascontiguousarray(origin, dtype=np.float64), ]
        self.d = [np.ascontiguousarray(direction / norm_2d(direction), dtype=np.float64), ]

        self.int_times = [t0, ]
        self.freq = [freq, ]
        # self.nfft = len(freq)
        # Share fft_freq across rays — compute only if not provided
        _fft_freq = kwargs.get('_fft_freq', None)
        if _fft_freq is not None and len(_fft_freq) == len(freq):
            self.fft_freq = _fft_freq
        else:
            self.fft_freq = rfftfreq(len(t), d=(t[-1]-t[0])/len(t))[:len(freq)]

        # Dispersion curve and phase coefficient for the birth direction:
        # f_shifted = exp(_phase_coeff * distance) * f
        self.fft_speed, self._phase_coeff = self._dispersion_for_direction(self.d[0])

        self.alive = True

        # Diffraction bookkeeping (see the class docstring / ray_diff)
        self.x_src = float(kwargs.get('x_src', 0.))
        self.dtheta = float(kwargs.get('dtheta', 0.))
        self.amp_law = kwargs.get('amp_law', None)
        self.diff_order = int(kwargs.get('diff_order', 0))
        self._vseen = None

        # Dominant frequency bin — invariant because fshift only rotates phases
        self._dom_freq_idx = np.argmax(np.abs(freq))
        self._dom_freq = self.fft_freq[self._dom_freq_idx]

        # Cache hash — computed once from the immutable initial state
        # (kind + direction + origin + birth time + spectrum). Array contents are
        # hashed through their raw bytes; building 500-element tuples was a
        # measurable cost per spawned ray.
        self._hash = hash((self.kind, self.d[0].tobytes(), self.trace_points[0].tobytes(),
                           float(self.int_times[0]), np.asarray(self.freq[0]).tobytes(),
                           kwargs.get('_hash_extra', None)))

    def amp_factor(self, x):
        """Amplitude factor of a diffracted ray at path position(s) ``x``.

        ``ray.a`` is law-free (the birth amplitude with damping only); the
        field of a diffracted fan is ``a * amp_factor(x)`` with
        ``amp_factor = rho/(X + rho) * T(w)``, ``rho = x_off + x`` the path
        length from the diffracting vertex, ``X`` the path from the origin of
        the incident family to the vertex (so that the fan, whose ray density
        is 1/rho, decays like the incident family, 1/(X + rho)), and ``T`` the
        UTD Fresnel transition (``utils_rays.utd.transition_T``) at the
        dominant frequency for the ray's angle ``dphi`` from the shadow
        boundary.  Returns 1 (scalar) for geometrical rays.
        """
        if self.amp_law is None:
            return 1.
        from utils_rays.utd import transition_T, fresnel_w
        X, dphi, k, x_off = self.amp_law
        rho = np.asarray(x, dtype=float) + x_off
        # UTD distance parameter of a point source, L = rho X / (rho + X)
        return rho / (X + rho) * transition_T(fresnel_w(k, rho * X / (X + rho), dphi))

    def _dispersion_for_direction(self, d) -> tuple:
        """Phase-velocity array and dispersion coefficient for direction ``d``.

        :param d: unit direction vector (2 components)
        :return: ``(fft_speed, phase_coeff)``; ``fft_speed[k]`` is the phase
                 velocity of FFT bin ``k`` (geometry length unit per second,
                 i.e. mm/s) and ``phase_coeff = -2j*pi*fft_freq/fft_speed`` so
                 that ``exp(phase_coeff * x) * f`` disperses spectrum ``f`` over
                 a distance ``x``.
        """
        medium = self.medium
        theta = wrap_angle_pi(np.arctan2(d[1], d[0]) + medium.theta)
        th_factor = medium.th / 1.E+6

        # Batch interpolation: single Numba call instead of n_fft Python round-trips
        if hasattr(medium.ws, 'batch_speed'):
            x_vals = np.ascontiguousarray(self.fft_freq * th_factor)
            fft_speed = medium.ws.batch_speed(self.kind, x_vals, theta) * 1.E3
        else:
            ws_func = getattr(medium.ws, self.kind)
            fft_speed = np.array([ws_func((fi * th_factor, theta)) * 1.E3
                                  for fi in self.fft_freq])

        # nan_to_num guards against 0/0 at the DC bin (fft_speed -> 0 at f = 0)
        phase_coeff = np.nan_to_num(
            (-0. - 1j) * 2 * np.pi * self.fft_freq / fft_speed,
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        return fft_speed, phase_coeff

    def _update_phase_coeff(self) -> None:
        """Recompute ``fft_speed`` and ``_phase_coeff`` for the current
        direction ``d[-1]``.

        Called by ``ray_refl`` after a reflection when
        ``dispersion_follows_direction`` is True, so that the segment that
        starts at the reflection point is dispersed with the curve of its own
        propagation angle. ``fft_speed``/``_phase_coeff`` therefore describe
        the *last* segment of the ray; use ``phase_coeff_at`` for earlier ones.
        """
        self.fft_speed, self._phase_coeff = self._dispersion_for_direction(self.d[-1])

    def phase_coeff_at(self, i: int) -> np.ndarray:
        """Dispersion coefficient valid on the segment that starts at event ``i``.

        During tracing the segment starting at event ``i`` was propagated with
        the coefficient of direction ``d[i]`` (updated at reflections only when
        ``dispersion_follows_direction`` is True). Signal reconstruction must
        use the same coefficient to stay consistent with the stored spectra.
        """
        if not dispersion_follows_direction:
            return self._phase_coeff
        d_i = self.d[i]
        if np.array_equal(d_i, self.d[-1]):
            return self._phase_coeff
        return self._dispersion_for_direction(d_i)[1]

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

        # Dominant frequency — cached, invariant across ray lifetime
        fi = self._dom_freq
        v = self.medium.v_ray(self, i, fi=fi)

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
            logging.debug('Ray {}: int_time - (trace ori, trace end):'.format(self) +
                          '\n'.join(['{:.2e} - ({:.2e}, {:.2e})'.format(w1, w2[0], w2[1])
                                     for w1, w2 in zip(self.int_times, self.trace_points)]))
            if x is not None:
                errormsg = 't value: {:.4e} smaller than last increment {}: {:.4e}, for x: {:.3e}'.format(t, i, t0, x)
            else:
                errormsg = 't value: {:.4e} smaller than last increment {}: {:.4e}'.format(t, i, t0)

            # raise TypeError(errormsg)
            logging.debug(errormsg)
            logging.debug('Ray {} is now dead, forever'.format(self))
            self.alive = False
            return x0, trace0, d, f0, a0, t

        trace_i = trace0 + d * x_i

        # Implement direction change?
        d_i = d

        # Dispersion and time shift
        f_i = self.medium.fshift(f0, x_i, self, i)

        # Transmission loss
        a_i = self.medium.tl(self, i, t - t0, fi=fi)

        return self.x[i] + x_i, trace_i, d_i, f_i, a_i, t

    def trace(self, t, r_map):
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

        # Find the NEAREST boundary crossed by the last trace segment: the
        # medium may be non-convex (e.g. a plate with the damage as a hole),
        # so several walls may be crossed and the order of medium.objs must
        # not matter.
        p0, p1 = self.trace_points[-2], self.trace_points[-1]
        best_s = None
        best = None
        for obj in self.medium.objs:
            h = obj.hit(p0, p1)
            if h is not None and (best_s is None or h[0] < best_s):
                best_s = h[0]
                best = (obj, h)

        # Sensors only see the trace up to that boundary: the part beyond it
        # is an overshoot that the reflection/refraction below discards.
        p_end = p1 if best is None else best[1][1]
        for sens in self.medium.sensors:
            sens.intersect(self, t, r_map, p_end)  # sensors don't interact

        # Edge diffraction: capture the vertices the (truncated) trace passes
        # within half a ray spacing, s = dtheta * (x_src + x).  Windows of
        # width s tile the wavefront, so exactly one ray per family captures
        # each vertex.  Rays born next to the vertex (fans, daughters spawned
        # on its faces) and rays that already diffracted there are excluded.
        if diffraction and self.dtheta > 0. and self.diff_order < diff_params['max_order'] \
                and self.medium.vertices:
            global _ray_diff
            if _ray_diff is None:
                from utils_rays.ray_utils import ray_diff as _rd
                _ray_diff = _rd
            seg_len = norm_2d(p_end - p0)
            for v in self.medium.vertices:
                if self._vseen is not None and id(v) in self._vseen:
                    continue
                dist, tf = _point_seg_dist_2d_numba(v.p, p0, p_end)
                x_foot = self.x[-2] + tf * seg_len
                half_s = 0.5 * self.dtheta * (self.x_src + x_foot)
                if dist <= half_s and norm_2d(p0 - v.p) > half_s:
                    if self._vseen is None:
                        self._vseen = []
                    self._vseen.append(id(v))
                    rfr_rays.extend(_ray_diff(self, v, x_foot, t, r_map))

        if best is not None:
            obj, (_, intersect, n, d) = best
            rfr_rays.extend(obj.interact(self, n, d, intersect, t, r_map))

        r_map.save_ray(self)  # saves ray
        return rfr_rays

    def retrace(self, length, r_map):
        """ Calculates additional n points inbetween traces

        # :param n: number of points
        :param length: "approximate" length for retracing
        :param r_map: ray map
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

        r_map.save_ray(self)  # saves ray

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
        s = a_i*self.amp_factor(x)*irfft(f_i, n=len(self.t))
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
        s = a_i*self.amp_factor(self.x[i])*irfft(f_i, n=len(self.t))
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
        return self._hash

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
            s = signal_f(self.t, 1.0, f, fd)

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
            # angular spacing of the fan: sets the vertex capture width
            self.dtheta = (theta_f - theta_i) / self.n_rays
            for i in range(self.n_rays):

                theta = (theta_f - theta_i) / self.n_rays * i + theta_d
                d = np.array([np.cos(theta_i + theta), np.sin(theta_i + theta)])

                if kind not in ['S0', 'A0']:
                    ray_a = Ray(origin=self.o, direction=d, freq=self.freq, t=self.t,
                                medium=medium, kind='A0', a=self.a0, nfft=self.nfft,
                                _fft_freq=self.fft_freq, dtheta=self.dtheta)
                    ray_s = Ray(origin=self.o, direction=d, freq=self.freq, t=self.t,
                                medium=medium, kind='S0', a=self.a0, nfft=self.nfft,
                                _fft_freq=self.fft_freq, dtheta=self.dtheta)
                    self.rays.append(ray_a)
                    self.rays.append(ray_s)
                else:
                    ray_i = Ray(origin=self.o, direction=d, freq=self.freq, t=self.t,
                                medium=medium, kind=kind, a=self.a0, nfft=self.nfft,
                                _fft_freq=self.fft_freq, dtheta=self.dtheta)
                    self.rays.append(ray_i)

    def inp_signal(self):
        return irfft(self.freq, n=len(self.t))*self.a0


class Beam_from_pzt(Beam):
    def __init__(self, n_rays, pzt,
                 signal_f=burst_hann, freq=None,
                 power=1., kind='all', **kwargs):

        self.source = pzt
        params = [pzt.origin(), ]
        medium = pzt.medium

        Beam.__init__(self, n_rays=n_rays, params=params, medium=medium,
                      signal_f=signal_f, freq=freq,
                      power=power, kind=kind, b_type='circ', **kwargs)
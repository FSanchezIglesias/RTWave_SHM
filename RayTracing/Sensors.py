from geom.objects_2d import Segment, Circunf, _SEG_TOL
from geom.geom_utils import (dot_2d, _seg_seg_intersect_2d_numba,
                             _circunf_seg_intersect_2d_numba)
import numpy as np
from scipy.fft import irfft
from scipy.signal.windows import hamming
import logging

_CIRC_TANGENT_TOL = 1.e-9


class _StrBoundary(Segment):
    def intersect_sens(self, ray, p1=None):
        # direct kernel call (trace points and endpoints are float64 arrays)
        p0 = ray.trace_points[-2]
        if p1 is None:
            p1 = ray.trace_points[-1]
        intersect = _seg_seg_intersect_2d_numba(p0, p1,
                                                self._a1, self._a2, _SEG_TOL)
        if intersect[0] != intersect[0]:  # NaN -> no intersection
            return None

        # t_int = norm_2d(intersect - ray.trace[-2]) /
        # norm_2d(ray.trace[-1] - ray.trace[-2]) * (t - ray.int_times[-2]) \
        #         + ray.int_times[-2]
        x_int = ray.x[-2] + dot_2d(intersect - p0, ray.d[-2])

        return [x_int, ]  # return as a list for compatibility


class _CircBoundary(Circunf):
    def intersect_sens(self, ray, p1=None):
        # direct kernel call; returns an (n, 2) array with n in {0, 1, 2}
        p0 = ray.trace_points[-2]
        if p1 is None:
            p1 = ray.trace_points[-1]
        intersect = _circunf_seg_intersect_2d_numba(self._c, self._r, p0, p1,
                                                    False, _CIRC_TANGENT_TOL)
        if intersect.shape[0] == 0:
            return None

        # t_int = norm_2d(intersect - ray.trace[-2]) / norm_2d(ray.trace[-1] - ray.trace[-2]) *
        # (t - ray.int_times[-2]) \
        #         + ray.int_times[-2]
        d0 = ray.d[-2]
        x0 = ray.x[-2]

        # Distance travelled over ray
        return [x0 + dot_2d(intersect[i] - p0, d0) for i in range(intersect.shape[0])]


class Sensor:

    def __init__(self, kind, params, sensitivity=1., name=None,
                 color='red', **kwargs):
        """
        Sensor is for now a rectangle defined by 4 corners
        :param kind: string:
                - 'rect' - Rectangle
                - 'sq' - Square
                - 'circ' - circle
        :param params: parameters to define the sensor boundary:
                - rectangle: 4 points [[a1, a2], [b1, b2], [c1, c2], [d1, d2]] ordered in a rhs motion
                - square: center and radius [[o1, o2], r] -> to form a square tho...
                - circle: center and radius [[o1, o2], r]
        """
        self.bounds = []
        self.kind = kind.lower()

        if name is not None:
            self.name = name
        else:
            import names
            self.name = names.get_full_name()

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

        self.signal_s = None

    def intersect(self, ray, t, h5file, p1=None):
        """ Checks for intersections but rays are not altered
        :param ray: Ray object
        :param t: maintains sig of other intersect methods
        :param h5file: maintains sig of other intersect methods
        :param p1: end point of the trace to check (default: the ray's last
            trace point).  ``Ray.trace`` passes the point where the trace hits
            the nearest boundary so that the overshooting part of the trace
            (beyond the wall the ray reflects/refracts on) records no crossing.
        :return: empty list
        """

        for b in self.bounds:
            int_points = b.intersect_sens(ray, p1)
            if int_points is not None:
                if ray.__hash__() in self.int_rays.keys():
                    for xi in int_points:
                        self.int_rays[ray.__hash__()].append(xi)
                else:
                    self.int_rays[ray.__hash__()] = int_points

        return []

    def _signal_on_ray(self, ray, xs_ray, d_x, window=None):
        """Integrate the signal carried by ``ray`` over its crossings of the sensor.

        ``xs_ray`` holds the ray-path coordinates at which the ray crossed the
        sensor boundary; they are sorted and taken in (entry, exit) pairs. Each
        chord is sampled every ``d_x`` (mid-point rule) and weighted by
        ``window``.

        The semantics are those of the original per-point implementation
        (``Ray.signal_at_x``): every integration point uses the ray event
        (segment) it lies in, points at or beyond the last recorded ray
        position are skipped, and the dispersion coefficient is the one of that
        segment's direction (``Ray.phase_coeff_at``). Within a run of
        consecutive points on the same segment the phase and damping factors
        are advanced by a step recurrence instead of re-evaluating the
        exponentials at every point.
        """
        xs_ray = sorted(set(xs_ray))
        n_t = len(ray.t)

        if len(xs_ray) < 2:
            return np.zeros(n_t)

        t_vec = ray.t
        medium = ray.medium
        x_arr = np.asarray(ray.x, dtype=float)
        x_last = x_arr[-1]

        total = np.zeros(n_t)

        for pair_idx in range(len(xs_ray) // 2):
            x_entry = xs_ray[2 * pair_idx]
            x_exit = xs_ray[2 * pair_idx + 1]

            xi = np.arange(x_entry, x_exit, d_x) + d_x / 2
            n_pts = xi.size
            if n_pts == 0:
                continue

            D_ray = abs(x_exit - x_entry)

            if window == 'hamming':
                w = hamming(n_pts)
            elif window == 'hsphere':
                w = hamming(n_pts)
                w *= D_ray / self.size
            else:
                w = np.ones(n_pts)
            w_dx = d_x * w

            # Ray event (segment) containing each point: last event with x <= xi.
            # Points at/after the ray's last position have no segment (v1 skipped them).
            seg_of = np.searchsorted(x_arr, xi, side='right') - 1
            valid = (seg_of >= 0) & (xi < x_last)
            n_skip = int(n_pts - valid.sum())
            if n_skip:
                logging.error('Unable to get signal for ray: {} at {} of {} points '
                              '(beyond ray end x = {:.3f})'.format(hash(ray), n_skip,
                                                                   n_pts, x_last))

            # Integrate runs of consecutive valid points lying on the same segment
            k = 0
            while k < n_pts:
                if not valid[k]:
                    k += 1
                    continue
                seg_idx = int(seg_of[k])
                k_end = k + 1
                while k_end < n_pts and valid[k_end] and seg_of[k_end] == seg_idx:
                    k_end += 1
                self._integrate_run(ray, seg_idx, xi[k:k_end], w_dx[k:k_end], d_x,
                                    t_vec, medium, total)
                k = k_end

        return total

    @staticmethod
    def _integrate_run(ray, seg_idx, xi, w_dx, d_x, t_vec, medium, total):
        """Accumulate into ``total`` the contribution of the points ``xi``.

        ``xi`` are uniformly spaced by ``d_x`` and all lie on ray segment
        ``seg_idx`` (the one starting at event ``seg_idx``). Phase and damping
        are advanced by a step recurrence:
        ``exp(alpha*(dx + d_x)) = exp(alpha*dx) * exp(alpha*d_x)``.
        """
        n_t = total.size
        if ray.amp_law is not None:
            w_dx = w_dx * ray.amp_factor(xi)   # diffracted ray: per-point amplitude law
        x0 = ray.x[seg_idx]
        f0 = ray.freq[seg_idx]
        a0 = ray.a[seg_idx]
        t0 = ray.int_times[seg_idx]
        v = medium.v_ray(ray, seg_idx)
        damping_rate = 2 * np.pi * ray._dom_freq * medium.xi / v
        phase_coeff = ray.phase_coeff_at(seg_idx)          # [n_fft] complex

        dx_k = xi - x0                                        # [n_pts]
        mask_idx = np.searchsorted(t_vec, (t0 + dx_k / v) / 2)  # [n_pts]

        step_phase = np.exp(phase_coeff * d_x)               # [n_fft], once per run
        step_amp = np.exp(-damping_rate * d_x)               # scalar, once per run

        cur_f = np.exp(phase_coeff * dx_k[0]) * f0           # [n_fft]
        cur_amp = a0 * np.exp(-damping_rate * dx_k[0])       # scalar

        # --- Exact reformulation of  sum_k mask_k( irfft(F_k) )  ---
        # The inverse FFT is linear and every mask is a prefix zeroing whose
        # index m_k is non-decreasing along the run.  Grouping points with the
        # same mask index (spectra S_g, masks m_g) and defining the cumulative
        # spectra C_g = S_0 + ... + S_g, the output sample n equals
        # irfft(C_{j(n)})[n] with j(n) = max{g : m_g <= n}.  Hence:
        #   * n <  m_0        -> 0
        #   * n >= m_{G-1}    -> irfft(C_{G-1})[n]      (one full transform)
        #   * m_0 <= n < m_{G-1} -> direct inverse-DFT evaluation of C_{j(n)}
        #                          at that single sample (n_fft terms each).
        # The transition window is only a few dozen samples long, so this
        # replaces one 10^4-point transform per point by one per run.
        n_pts = xi.size
        group_spectra = []
        group_masks = []
        acc = cur_amp * w_dx[0] * cur_f
        cur_mask = int(mask_idx[0])
        for k in range(1, n_pts):
            cur_f *= step_phase      # n_fft complex mults, no exp
            cur_amp *= step_amp      # 1 scalar mult
            if mask_idx[k] != cur_mask:
                group_spectra.append(acc)
                group_masks.append(cur_mask)
                acc = cur_amp * w_dx[k] * cur_f
                cur_mask = int(mask_idx[k])
            else:
                acc += cur_amp * w_dx[k] * cur_f
        group_spectra.append(acc)
        group_masks.append(cur_mask)

        cum = np.cumsum(np.array(group_spectra), axis=0)     # [n_groups, n_fft]
        m_first, m_last = group_masks[0], group_masks[-1]

        # Full transform of the total spectrum, valid from the last mask on
        full = irfft(cum[-1], n=n_t)
        if m_last < n_t:
            total[m_last:] += full[m_last:]

        # Transition window: direct evaluation of irfft(C_{j(n)})[n]
        if m_last > m_first:
            n_win = np.arange(m_first, min(m_last, n_t))
            masks = np.asarray(group_masks)
            j_of_n = np.searchsorted(masks, n_win, side='right') - 1
            n_fft = cum.shape[1]
            kbins = np.arange(n_fft)
            # scipy.fft.irfft convention: y[n] = (1/N) [Re x_0 + 2 sum_{k>=1} Re(x_k e^{+2 pi i k n / N})]
            # (bins beyond n_fft are zero; n_fft < N/2 so there is no Nyquist term)
            phase = np.exp((2j * np.pi / n_t) * np.outer(n_win, kbins))   # [n_win, n_fft]
            weights = np.full(n_fft, 2.0)
            weights[0] = 1.0
            vals = (cum[j_of_n] * phase * weights).real.sum(axis=1) / n_t
            total[n_win] += vals

    def signal(self, d_x=0.1, procs=None, window='hsphere'):
        """ Measure signal at sensor
        Considers only rays that cut twice
        :param t: time points
        :param d_x: integration step
        :param procs: number of proc for parallel
        :param window: integration window for ray power function, default 'hsphere'
        :return: measured signal
        """

        if not self.int_rays:
            logging.info('No rays intersecting sensor {}'.format(self.name))
            return None

        # Time must be equal for all model, its taken from map
        t = self.map.t
        signal_mat = np.zeros([len(t), len(self.int_rays)])

        logging.info('Calc signal on sensor {}. Integrating over {} rays'.format(self.name, len(self.int_rays)))

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
        self.signal_s = signal * self.sensitivity

        return self.signal_s

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
            raise TypeError('Sensor:{} already has a medium defined.'.format(self))

    def get_limits(self):
        """
        Limits of square definition on x and y
        """
        xmax, xmin, ymax, ymin = None, None, None, None

        for obj in self.bounds:
            xmax_o, xmin_o, ymax_o, ymin_o = obj.get_limits()
            if xmax is None:
                xmax, xmin, ymax, ymin = xmax_o, xmin_o, ymax_o, ymin_o
            else:
                xmax = xmax_o if xmax_o > xmax else xmax
                xmin = xmin_o if xmin_o < xmin else xmin
                ymax = ymax_o if ymax_o > ymax else ymax
                ymin = ymin_o if ymin_o < ymin else ymin

        return xmax, xmin, ymax, ymin
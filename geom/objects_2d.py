import numpy as np

# import math
from geom.geom_utils import (seg_seg_intersect_2d, norm_2d, cross_2d, dot_2d, _seg_seg_intersect_2d_numba,
                             _ellipse_seg_intersect_2d_numba, _point_seg_dist_2d_numba, wrap_angle_pi)
from utils_rays.ray_utils import ray_refl, ray_refr
import RayTracing.Ray as _ray_module

_SEG_TOL = 1.e-9  # distance tolerance [mm] of the segment-segment intersection test


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

        # Pre-cached lookups for hot-path (v_ray / tl)
        self._th_factor = th / 1.E+6
        self._ws_func = {'S0': ws.S0, 'A0': ws.A0}

        # List of objects contained in the medium
        self.objs = []
        self.sensors = []
        # Diffracting vertices (corners) of the medium's walls, filled by
        # ``build_vertices`` (called from ``Map2D.__init__``)
        self.vertices = []

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
            if hasattr(obj, 'signal'):
                self.sensors.append(obj)
            else:
                self.objs.append(obj)
            if hasattr(obj, 'add_medium'):
                obj.add_medium(self)

    def v_ray(self, ray, i=-1, fi=None):
        if fi is None:
            fi = ray._dom_freq
        f_d = fi * self._th_factor
        theta = np.arctan2(ray.d[i][1], ray.d[i][0]) + self.theta
        theta = wrap_angle_pi(theta)

        return self._ws_func[ray.kind]((f_d, theta)) * 1.E3

    def v_dir(self, kind, fi, d):
        """Phase velocity [mm/s] of mode ``kind`` at frequency ``fi`` [Hz]
        for the unit direction ``d`` (``v_ray`` without a ``Ray``)."""
        theta = wrap_angle_pi(np.arctan2(d[1], d[0]) + self.theta)
        return self._ws_func[kind]((fi * self._th_factor, theta)) * 1.E3

    def phase_coeff(self, kind, fft_freq, d):
        """Dispersion coefficient ``-2j pi f / v(f)`` of mode ``kind`` for the
        unit direction ``d`` (``Ray._dispersion_for_direction`` without a
        ``Ray``): ``exp(phase_coeff * x) * F`` propagates a spectrum ``F``
        over ``x`` mm in this medium."""
        theta = wrap_angle_pi(np.arctan2(d[1], d[0]) + self.theta)
        if hasattr(self.ws, 'batch_speed'):
            x_vals = np.ascontiguousarray(fft_freq * self._th_factor)
            v = self.ws.batch_speed(kind, x_vals, theta) * 1.E3
        else:
            v = np.array([self._ws_func[kind]((fi * self._th_factor, theta)) * 1.E3
                          for fi in fft_freq])
        return np.nan_to_num((-0. - 1j) * 2 * np.pi * fft_freq / v,
                             nan=0.0, posinf=0.0, neginf=0.0)
    
    # def tl(self, ray):
    #     # the ray has already advanced
    #     t = ray.int_times[-1] - ray.int_times[-2]
    #     
    #     a_damp = ray.a[-1] * np.exp(-2*np.pi*ray.freq[-1]*self.xi*t)
    #     return a_damp
    
    def tl(self, ray, i, t, fi=None):
        # the ray has already advanced
        # Rays only stay on a single medium
        if fi is None:
            fi = ray._dom_freq
        a_damp = ray.a[i] * np.exp(-2*np.pi*fi*self.xi*t)
        return a_damp

    def fshift_dispersion(self, f, x, ray, i=-1):
        """ Dispersion estimation based on the FFT shift property

        :param f: X components of the fourier transform
        :param x: Distance
        :returns f_d: X components of the signal disperse
        """

        f_d = np.exp(ray._phase_coeff * x) * f
        return f_d

    def fshift_nd(self, f, x, ray, i=-1):
        """ fft shift assuming average/non-dispersive velocity. """

        v = self.v_ray(ray, i)
        f_d = np.exp((0. - 1j) * 2 * np.pi * ray.fft_freq * x/v) * f
        return f_d

    def __hash__(self):
        return self.__hash

    def get_limits(self):
        """
        Limits of square medium definition on x and y
        """
        xmax, xmin, ymax, ymin = None, None, None, None

        for obj in self.objs:
            xmax_o, xmin_o, ymax_o, ymin_o = obj.get_limits()
            if xmax is None:
                xmax, xmin, ymax, ymin = xmax_o, xmin_o, ymax_o, ymin_o
            else:
                xmax = xmax_o if xmax_o > xmax else xmax
                xmin = xmin_o if xmin_o < xmin else xmin
                ymax = ymax_o if ymax_o > ymax else ymax
                ymin = ymin_o if ymin_o < ymin else ymin

        return xmax, xmin, ymax, ymin


def _interact(obj, ray, n, d, intersect, t, map):
    """Reflect/refract ``ray`` at the boundary ``obj`` hit at ``intersect``.

    Shared by every boundary primitive that rays interact with (``Segment``,
    ``Ellipse``).  ``obj`` must expose ``mediums``, ``ratio_rfl``,
    ``ratio_mode`` and ``bl``.  ``n`` is the unit normal pointing away from
    the side the ray comes from and ``d`` a unit tangent.

    :return: hashes of the new rays spawned at the boundary
    """
    irays = []

    # estimate intersc. time
    t_int = norm_2d(intersect-ray.trace_points[-2]) / norm_2d(ray.trace_points[-1]-ray.trace_points[-2]) \
            * (t-ray.int_times[-2]) + ray.int_times[-2]

    # Snell's law for every medium behind the wall, with the incident
    # direction (before the reflection below mutates it).  Total internal
    # reflection: no medium can take the transmitted share -> it is reflected.
    others = [m2 for m2 in obj.mediums if m2 is not ray.medium]
    v_in = ray.medium.v_ray(ray) if others else None
    sin_i = dot_2d(ray.d[-1], d) if others else 0.
    v2_v1 = {id(m2): m2.v_ray(ray) / v_in for m2 in others}
    tir = bool(others) and all(abs(v2_v1[id(m2)] * sin_i) > 1. for m2 in others)
    ratio_rfl = 1. if (tir and _ray_module.total_internal_reflection) else obj.ratio_rfl

    # Reflect ray
    irays_rfl, ray_params_i = ray_refl(ray, n, d, intersect, t_int, t,
                                       ratio=ratio_rfl, ratio_mode=obj.ratio_mode,
                                       bl=obj.bl, map=map)
    irays.extend(irays_rfl)

    # Refract ray on all remaining boundaries
    if others and not tir:
        x_i, trace_i, d_i, f_i, a_i, t_i = ray_params_i
        ratio_rfr = (1 - obj.ratio_rfl) / len(others)
        for m2 in others:
            irays.extend(ray_refr(ray, n, d, intersect, t_int, t, d_i, a_i, f_i,
                                  ratio=ratio_rfr, m2=m2, ratio_mode=obj.ratio_mode,
                                  bl=obj.bl, map=map, v2_v1=v2_v1[id(m2)]))

    return irays


class Segment:
    def __init__(self, a1, a2, boundary_losses=0.2,
                 color='black',
                 B=np.array([[1,0,0], [0,1,0]]), P=np.zeros(3),
                 ratio_rfl=0.8, ratio_mode=0.9):

        self.a1 = a1
        self.a2 = a2
        # contiguous float64 copies for direct calls into the Numba kernels
        self._a1 = np.ascontiguousarray(a1, dtype=np.float64)
        self._a2 = np.ascontiguousarray(a2, dtype=np.float64)

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

    def hit(self, p0, p1):
        """First crossing of the trace ``p0 -> p1`` with this wall.

        :return: ``None`` if the trace does not cross the wall, otherwise
            ``(s, intersect, n, d)`` with ``s`` the distance of the crossing from
            ``p0`` [mm], ``n`` the unit normal oriented AWAY from the side the
            trace comes from (into the medium behind the wall) and ``d`` the
            matching unit tangent.
        """
        # direct kernel call: trace points and endpoints are float64 arrays
        intersect = _seg_seg_intersect_2d_numba(p0, p1, self._a1, self._a2, _SEG_TOL)
        if intersect[0] != intersect[0]:  # NaN -> no intersection
            return None

        # self.n is the right-hand normal of self.d, and cross > 0 means the
        # trace starts on the left of the wall.
        if cross_2d(self.d, p0 - self._a1) > 0:
            n, d = self.n, self.d
        else:
            n, d = -self.n, -self.d

        return norm_2d(intersect - p0), intersect, n, d

    def interact(self, ray, n, d, intersect, t, map):
        """Reflect/refract ``ray`` at the crossing found by ``hit`` (see ``_interact``)."""
        return _interact(self, ray, n, d, intersect, t, map)

    def intersect(self, ray, t, map):
        """Intersect the last trace of ``ray`` with self and reflect/refract it.

        Kept for external callers; ``Ray.trace`` uses ``hit`` on every object of
        the medium and interacts only with the nearest crossing.

        :param ray: ray that intersects
        :param t: time of analysis
        :return: reflected/refracted new rays
        """
        h = self.hit(ray.trace_points[-2], ray.trace_points[-1])
        if h is None:
            return []
        _, intersect, n, d = h
        return _interact(self, ray, n, d, intersect, t, map)

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

    def get_limits(self):
        xmax = max([self.a1[0], self.a2[0]])
        xmin = min([self.a1[0], self.a2[0]])
        ymax = max([self.a1[1], self.a2[1]])
        ymin = min([self.a1[1], self.a2[1]])

        return xmax, xmin, ymax, ymin


class Ellipse:
    def __init__(self, c, a, b, phi=0., boundary_losses=0.2,
                 color='black',
                 B=np.array([[1, 0, 0], [0, 1, 0]]), P=np.zeros(3),
                 ratio_rfl=0.8, ratio_mode=0.9):
        """Elliptical boundary that rays reflect on / refract through.

        Same interface as ``Segment`` (``add_medium``, ``intersect``, ``plot``,
        ``get_limits``) so it can be added to a ``medium`` like any wall.  A
        circular boundary is ``Ellipse(c, r, r)`` (``Circunf`` is plot/sensor
        only and does not interact with rays).

        The ellipse may be a hole in a larger (non-convex) medium: ``Ray.trace``
        interacts with the nearest crossing among all the objects of the medium,
        so the order of ``objs`` does not matter.

        :param c: centre [mm]
        :param a: semi-axis along the local x direction [mm]
        :param b: semi-axis along the local y direction [mm]
        :param phi: rotation of the local x axis w.r.t. the global one [rad]
        """
        self.c = np.asarray(c, dtype=float)
        self.a = float(a)
        self.b = float(b)
        self.phi = float(phi)
        # contiguous float64 copies / scalars for direct calls into the Numba kernel
        self._c = np.ascontiguousarray(self.c, dtype=np.float64)
        self._cos = float(np.cos(self.phi))
        self._sin = float(np.sin(self.phi))

        self.color = color

        # List of mediums that contain the boundary
        self.mediums = []

        # energy factors
        self.ratio_rfl = ratio_rfl
        self.ratio_mode = ratio_mode
        self.bl = boundary_losses

        self.B = B
        self.P = P

    def add_medium(self, medium):
        """ Adds medium to list
        """
        self.mediums.append(medium)

    def contains(self, p):
        """True if the point ``p`` lies inside the ellipse."""
        v = np.asarray(p, dtype=float) - self.c
        u = (self._cos * v[0] + self._sin * v[1]) / self.a
        w = (-self._sin * v[0] + self._cos * v[1]) / self.b
        return u * u + w * w < 1.

    def hit(self, p0, p1):
        """First crossing of the trace ``p0 -> p1`` with the ellipse (see ``Segment.hit``)."""
        res = _ellipse_seg_intersect_2d_numba(self._c, self.a, self.b, self._cos, self._sin,
                                              p0, p1, _SEG_TOL)
        if res[0] != res[0]:  # NaN -> no intersection
            return None

        intersect = res[:2].copy()
        n_out = res[2:4].copy()
        # n must point away from the side the trace comes from (see Segment):
        # outward when it comes from inside, inward otherwise
        n = n_out if res[4] > 0.5 else -n_out
        d = np.array([-n[1], n[0]])
        return norm_2d(intersect - p0), intersect, n, d

    def interact(self, ray, n, d, intersect, t, map):
        """Reflect/refract ``ray`` at the crossing found by ``hit`` (see ``_interact``)."""
        return _interact(self, ray, n, d, intersect, t, map)

    def intersect(self, ray, t, map):
        """Intersect the last trace of ``ray`` with self and reflect/refract it
        (see ``Segment.intersect``)."""
        h = self.hit(ray.trace_points[-2], ray.trace_points[-1])
        if h is None:
            return []
        _, intersect, n, d = h
        return _interact(self, ray, n, d, intersect, t, map)

    def plot(self, ax, color='default', marker=None):
        if color == 'default':
            color = self.color
        if color is None:
            return

        from matplotlib.patches import Ellipse as _MplEllipse
        c = self.B.T.dot(self.c) + self.P
        e = _MplEllipse((c[0], c[1]), 2 * self.a, 2 * self.b,
                        angle=np.degrees(self.phi), color=color, fill=False, zorder=2)
        ax.add_patch(e)

        if marker is not None:
            ax.plot(c[0], c[1], marker=marker, color=color)

    def get_limits(self):
        """Axis-aligned bounding box of the rotated ellipse."""
        hx = (self.a ** 2 * self._cos ** 2 + self.b ** 2 * self._sin ** 2) ** 0.5
        hy = (self.a ** 2 * self._sin ** 2 + self.b ** 2 * self._cos ** 2) ** 0.5
        return self.c[0] + hx, self.c[0] - hx, self.c[1] + hy, self.c[1] - hy


class Circunf:
    def __init__(self, c, r, color='black'):
        self.c = c
        self.r = r
        # contiguous float64 copies for direct calls into the Numba kernels
        self._c = np.ascontiguousarray(c, dtype=np.float64)
        self._r = float(r)

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

    def get_limits(self):
        """
        Limits of square medium definition on x and y
        """
        return self.c[0]+self.r, self.c[0]-self.r, self.c[1]+self.r, self.c[1]-self.r


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

_VERTEX_TOL = 1.e-6  # [mm] endpoints closer than this are the same vertex


class Vertex:
    """A corner where the endpoints of one or two ``Segment`` walls meet.

    Vertices are secondary sources of diffracted rays (see
    ``utils_rays.ray_utils.ray_diff``).  A vertex with a single face is a free
    edge (e.g. the tip of a screen inside a medium); one with two faces is a
    wedge.  The angular sector that a given medium occupies at the vertex is
    resolved when a ray is captured (``sector``), because the walls do not
    know on which side each medium lies.

    :param p: position [mm]
    :param faces: list of 1 or 2 ``Segment`` ending at ``p``
    """

    def __init__(self, p, faces, walls=None):
        self.p = np.ascontiguousarray(p, dtype=np.float64)
        self.faces = list(faces)
        # walls of the obstacle the vertex belongs to (the faces plus every
        # segment connected to them through shared endpoints); the local
        # probe tracing of ``ray_diff`` interacts with these only
        self.walls = list(faces) if walls is None else list(walls)
        # unit tangents from the vertex towards the other endpoint of each face
        self.tangents = []
        for f in self.faces:
            other = f.a2 if norm_2d(f.a1 - self.p) < norm_2d(f.a2 - self.p) else f.a1
            t = np.asarray(other, dtype=np.float64) - self.p
            self.tangents.append(t / norm_2d(t))
        self.mediums = []
        for f in self.faces:
            for m in f.mediums:
                if m not in self.mediums:
                    self.mediums.append(m)

    @property
    def single_medium(self):
        """True if every face belongs to exactly one, common, medium."""
        return all(len(f.mediums) == 1 for f in self.faces) and len(self.mediums) == 1

    def sector(self, r):
        """Angular sector of the vertex that contains the direction ``r``.

        :param r: direction from the vertex towards a point known to lie in
            the medium of interest
        :return: ``(faces, tangents, n_out, a_start, extent)``: the faces
            ordered (start, end) counter-clockwise, their tangents, the
            unit normals of each face pointing OUT of the sector, the start
            angle [rad] and the counter-clockwise extent [rad] of the sector.
        """
        if len(self.faces) == 1:
            t = self.tangents[0]
            f = self.faces[0]
            if cross_2d(t, r) > 0:      # r counter-clockwise of the face
                n_out = np.array([t[1], -t[0]])     # clockwise normal
            else:
                n_out = np.array([-t[1], t[0]])
            a0 = np.arctan2(t[1], t[0])
            return [f, f], [t, t], [n_out, n_out], a0, 2. * np.pi

        tA, tB = self.tangents
        aA = np.arctan2(tA[1], tA[0])
        aB = np.arctan2(tB[1], tB[0])
        ar = np.arctan2(r[1], r[0])
        ext_AB = (aB - aA) % (2. * np.pi)
        if (ar - aA) % (2. * np.pi) < ext_AB:
            fs, fe, ts, te, a0, ext = self.faces[0], self.faces[1], tA, tB, aA, ext_AB
        else:
            fs, fe, ts, te, a0, ext = self.faces[1], self.faces[0], tB, tA, aB, 2. * np.pi - ext_AB
        # the sector lies counter-clockwise of the start face and clockwise of
        # the end face; normals pointing out of the sector:
        n_s = np.array([ts[1], -ts[0]])
        n_e = np.array([-te[1], te[0]])
        return [fs, fe], [ts, te], [n_s, n_e], a0, ext

    def __repr__(self):
        return 'Vertex({:.3f}, {:.3f}; {} face(s))'.format(self.p[0], self.p[1], len(self.faces))


def build_vertices(mediums, tol=_VERTEX_TOL):
    """Find the diffracting vertices of the ``Segment`` walls of ``mediums``
    and store them in ``medium.vertices`` (a vertex is listed in every medium
    that contains one of its faces).

    Endpoints shared by two segments form a wedge vertex; an endpoint that
    belongs to a single segment and does not lie on another segment is a free
    edge.  Endpoints shared by more than two segments (e.g. the legacy cell
    mesh with invisible walls) are skipped with a warning: diffraction is only
    supported on the hole mesh.  ``Ellipse`` boundaries have no vertices.
    """
    import logging
    segs = []
    for m in mediums:
        for o in m.objs:
            if isinstance(o, Segment) and o not in segs:
                segs.append(o)
    pts = []       # list of [point, [segments]]
    for sg in segs:
        for e in (sg._a1, sg._a2):
            for entry in pts:
                if norm_2d(entry[0] - e) < tol:
                    if sg not in entry[1]:
                        entry[1].append(sg)
                    break
            else:
                pts.append([e.copy(), [sg]])

    # connected components of the walls (shared endpoints) = obstacles
    adj = {id(sg): set() for sg in segs}
    for _, fs in pts:
        for f in fs:
            for g in fs:
                if f is not g:
                    adj[id(f)].add(id(g))
    by_id = {id(sg): sg for sg in segs}
    comp_of = {}
    for sg in segs:
        if id(sg) in comp_of:
            continue
        group = []
        comp_of[id(sg)] = group
        stack = [id(sg)]
        while stack:
            cur = stack.pop()
            group.append(by_id[cur])
            for nb in adj[cur]:
                if nb not in comp_of:
                    comp_of[nb] = group
                    stack.append(nb)

    # closed components whose walls all belong to one medium and span that
    # medium's bounding box are the medium's outer boundary (e.g. the plate
    # outline): their corners are interior right angles and do not diffract
    outer = set()
    for group in set(id(g) for g in comp_of.values()):
        walls = next(g for g in comp_of.values() if id(g) == group)
        meds = set(id(m) for w in walls for m in w.mediums)
        if len(meds) != 1 or any(len(w.mediums) != 1 for w in walls):
            continue
        med = walls[0].mediums[0]
        xs = [w._a1[0] for w in walls] + [w._a2[0] for w in walls]
        ys = [w._a1[1] for w in walls] + [w._a2[1] for w in walls]
        xmax, xmin, ymax, ymin = med.get_limits()
        if (abs(max(xs) - xmax) < tol and abs(min(xs) - xmin) < tol and
                abs(max(ys) - ymax) < tol and abs(min(ys) - ymin) < tol):
            outer.add(group)

    vertices = []
    for p, faces in pts:
        if id(comp_of[id(faces[0])]) in outer:
            continue
        if len(faces) > 2:
            logging.warning('Vertex at ({:.3f}, {:.3f}) shared by {} walls: '
                            'diffraction not supported there'.format(p[0], p[1], len(faces)))
            continue
        if len(faces) == 1:
            # free edge unless the point lies on another wall
            on_wall = False
            for sg in segs:
                if sg is faces[0]:
                    continue
                d, _ = _point_seg_dist_2d_numba(p, sg._a1, sg._a2)
                if d < tol:
                    on_wall = True
                    break
            if on_wall:
                continue
        vertices.append(Vertex(p, faces, walls=comp_of[id(faces[0])]))

    for m in mediums:
        m.vertices = [v for v in vertices if m in v.mediums]
    return vertices

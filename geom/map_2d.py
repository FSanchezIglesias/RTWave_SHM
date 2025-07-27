# import numpy as np
import logging

import numpy as np
from tqdm import tqdm
import h5py

import RayTracing.Sensors
from geom import objects_2d
from utils_rays.ray_utils import split_ray, load_ray, save_ray
import gc


class Map2D:
    def __init__(self, init_beam=None, mediums=(), h5_fname=None):

        if h5_fname is None:
            from tempfile import SpooledTemporaryFile
            h5_fname = SpooledTemporaryFile()
        # File is opened in init
        self.h5file = h5py.File(h5_fname, 'a')

        self.mediums = {m.__hash__(): m for m in mediums}
        self.sensors = []

        for m in mediums:
            for o in m.objs:
                if hasattr(o, 'map'):
                    o.map = self  # store reference to self on objects that may need it
                if isinstance(o, RayTracing.Sensors.Sensor):
                    self.sensors.append(o)
        # self.t_solved = 0.

        self.init_beam = None
        self.t = [0., 0.2, 0.4, 0.6, 0.8, 1.]
        self.rays_h = []

        if init_beam is not None:
            self.set_init_beam(init_beam)

    def set_init_beam(self, init_beam):
        self.init_beam = init_beam
        self.t = init_beam.t

        # Save the rays from the initial beam
        for r in self.init_beam.rays:
            self.save_ray(r)
        self.rays_h = [n.__hash__() for n in self.init_beam.rays]  # store the hash keys of the rays

    def calc_t(self, t=None, procs=None):
        t = self.t.max() if t is None else t

        # if t <= self.t_solved:
        #    return
        # import numpy as np
        # incs = np.linspace(self.t_solved, t, nincs+1)[1:]

        # o_rays = [r for r in self.rays]  # copy the original rays to propagate
        # Propagate all rays a time t
        if (procs is None) or (procs == 1):
            for i in tqdm(range(len(self.rays_h))):
                ray = self.get_ray(self.rays_h[i])
                rays_r = self.trace_ray(ray, t)  # returns hashes
                self.rays_h += rays_r  # new rays are appended always at the end

        else:
            raise NotImplementedError

            from multiprocessing import Pool
            p = Pool(procs)
            r = []

            for rhash in self.rays_h:
                ray = self.get_ray(rhash)
                args = [self, ray, t]
                r.append(p.apply_async(self.trace_ray, args))

            rays_r = [res.get() for res in r]
            for rays_h in rays_r:
                self.rays_h += rays_h
        # self.t_solved = t

    def trace_ray(self, ray, t):
        """ Traces a ray included in map
        """
        rays_r = ray.trace(t, self)  # returns hashes
        gc.collect()
        return rays_r

    def calc_iter(self, N, t=None, procs=None):
        """ Calculates the map up to t in N iterations
        """
        t = self.t.max() if t is None else t
        for i in range(N):
            self.calc_t(t=i*t/N, procs=procs)

    def calc_signal(self, calc_source=False):
        """
        Calculates the signal of all sensors from the map
        """
        for s in self.sensors:
            if not calc_source and s == self.init_beam.source:
                print('ignoring sensor {}'.format(s.name))
                continue

            s.signal()

    def retrace_rays(self, length):
        """ Executes the retrace method on all rays stored in the map
        """
        logging.info('Retracing {} rays for a length of: {:.2e}'.format(len(self.rays_h), length))
        for i in tqdm(range(len(self.rays_h))):
            ray = self.get_ray(self.rays_h[i])
            ray.retrace(length, map=self)

    def plot2d(self, ax=None, marker=None, ray_color=None, ray_norm='norm', ray_linestyle='-'):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(16, 9))

        # norm value for rays -  taken from orig beam object
        if ray_norm == 'norm':
            ray_norm = self.init_beam.a0

        for rh in self.rays_h:

            try:
                ray = self.get_ray(rh)
                ray.plot(ax, marker=marker, color=ray_color, norm=ray_norm, linestyle=ray_linestyle)
            except TypeError:
                logging.error('Unable to load ray: {: #X}'.format(rh))

        # # Origin
        # ax.scatter(self.o[0], self.o[1])

        # Segments
        if self.mediums is not None:
            for m in self.mediums.values():
                for o in m.objs:
                    o.plot(ax, marker=marker)

        ax.set_aspect('equal')

        return ax

    def plot3d(self, ax=None, marker=None, ray_color=None, ray_norm='norm'):
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(projection='3d')

        # norm value for rays -  taken from orig beam object
        if ray_norm == 'norm':
            ray_norm = self.init_beam.a0

        for rh in self.rays_h:
            ray = self.get_ray(rh)
            ray.plot3d(ax, marker=marker, color=ray_color, norm=ray_norm)

        # # Origin
        # ax.scatter(self.o[0], self.o[1])

        # Segments
        if self.mediums is not None:
            for m in self.mediums.values():
                for o in m.objs:
                    if hasattr(o, 'plot3d'):
                        o.plot3d(ax, marker=marker)

        ax.set_aspect('equal')

        return ax

    def plot2d_contour(self, ax=None, gridlen=2., t_ind=-1, kind=None,
                       bounds=None, lvl_lim=None, lvl_N=22,
                       # err_val=0.001,
                       Nan_zero=True, border_lim=0.
                       ):
        """

        :param ax: matplotlib ax object
        :param gridlen: lenght grid divisions
        :param t_ind: time index
        :param kind: ray kind: 'A0', 'S0', None
        :param bounds: Grid bounds, list [xmin, xmax, ymin, ymax]
        :param lvl_lim:
        :param err_val: ??
        :param Nan_zero: Plot in white the areas where no ray is computed
        :return:
        """
        import numpy as np

        if bounds is None:
            lcx = []
            lcy = []
            for m in self.mediums.values():
                for o in m.objs:
                    if isinstance(o, objects_2d.Segment):
                        lcx.append(o.a1[0])
                        lcx.append(o.a2[0])
                        lcy.append(o.a1[1])
                        lcy.append(o.a2[1])
            xmin = min(lcx)
            xmax = max(lcx)

            ymin = min(lcy)
            ymax = max(lcy)

            xytol = border_lim*np.abs(xmax-xmin+ymax-ymin)
            xmin -= xytol
            xmax += xytol
            ymin -= xytol
            ymax += xytol

        else:
            xmin, xmax, ymin, ymax = bounds

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(16, 9))

        ngridx = int(np.abs(xmax-xmin)/gridlen)
        ngridy = int(np.abs(ymax-ymin)/gridlen)
        x_axis = np.linspace(xmin, xmax, ngridx)
        y_axis = np.linspace(ymin, ymax, ngridy)
        X, Y = np.meshgrid(x_axis, y_axis)

        args = [t_ind,
                xmin, xmax, ngridx,
                ymin, ymax, ngridy,
#                err_val
                ]

        rays_z = []
        z_val = np.zeros(X.T.shape)
        zi_val = np.zeros(X.T.shape)
        # for rh in self.rays_h:
        for i in tqdm(range(len(self.rays_h))):
            rh = self.rays_h[i]
            ray = self.get_ray(rh)
            if kind is not None:
                if kind != ray.kind:
                    continue
            # rays_z.append(split_ray(ray, *args))  #  for r in m.rays if r.kind != 'A0']
            z_ray, zi_ray = split_ray(ray, *args)
            z_val += z_ray
            zi_val += zi_ray
        # z_val = sum(rays_z)
        if lvl_lim is None:
            lvl_lim = np.max(np.abs(z_val))

        # Make Nans the points where no ray is computed
        if Nan_zero is True:
            z_val[zi_val == 0] = np.NaN

        levels = list(lvl_lim/2*np.linspace(-1, 1, lvl_N)**9)
        ax.contourf(X, Y, z_val.T, vmin=-np.max(lvl_lim)*.6, vmax=np.max(lvl_lim)*.6, levels=levels)

        if self.mediums is not None:
            for m in self.mediums.values():
                for o in m.objs:
                    o.plot(ax)
        ax.set_aspect('equal')
        return ax, z_val, zi_val, [X, Y]

    def to_vtk(self, filename, version='2.0', dtype='ASCII'):
        """ Writes paraview vtk file

        :param filename: VTK filename
        :param version: Paraview vtk version file, only 2.0 supported
        :return:
        """
        # TODO: Finish this
        raise NotImplementedError('vtk file conversion not implemented')

        if version != '2.0':
            raise TypeError('Unkown version: {}'.format(version))

        with open(filename, 'w') as vtk:

            vtk.write('# vtk DataFile Version {}'.format(version))
            vtk.write('')

    def close_h5(self):
        """ Closes hdf5 file """
        self.h5file.close()

    def get_ray(self, ray_h):
        return load_ray(ray_h, self.h5file, self)

    def save_ray(self, ray):
        save_ray(ray, self.h5file)

    def save_signals(self, fname, key='', format='hdf', use_pandas=False):
        """
        Saves the sensors signals on a file
        :param fname: Filename
        :param key: HDF file name path
        :param format: only hdf supported
        :return:
        """

        if format != 'hdf':
            raise TypeError('Unknown format {}'.format(format))

        sensors_s = []
        sensors_l = []
        for s in self.sensors:
            if s.signal_s is not None:
                sensors_s.append(s.signal_s)
                sensors_l.append(s.name)
        sensors_s = np.array(sensors_s).T

        if use_pandas:
            import pandas as pd
            sensors_signaldf = pd.DataFrame(sensors_s, index=self.t, columns=sensors_l)
            sensors_signaldf.to_hdf(fname, key=key)
        else:
            import h5py

            with h5py.File(fname, 'a') as h5f:
                if key in h5f:
                    raise KeyError('HDF5 file already contains a dataset with the same key: {}'.format(key))
                else:
                    h5f[key] = sensors_s
                    h5f[key].attrs['columns'] = sensors_l

    def add_sensor(self, sens):
        xmax_s, xmin_s, ymax_s, ymin_s = sens.get_limits()

        for i, m in self.mediums.items():
            xmax_m, xmin_m, ymax_m, ymin_m = m.get_limits()
            # x
            if xmax_m > xmax_s:
                if xmin_m < xmin_s:
                    if ymax_m > ymax_s:
                        if ymin_m < ymin_s:
                            # The sensor is only added to the first medium that matches
                            m.add_objs([sens, ])
                            if hasattr(sens, 'map'):
                                sens.map = self
                            self.sensors.append(sens)
                            return
        raise KeyError('Unable to add sensor: {}'.format(sens))

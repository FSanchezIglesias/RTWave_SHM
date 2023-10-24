# import numpy as np
import logging
from tqdm import tqdm
import h5py
from utils_rays.ray_utils import split_ray, load_ray, save_ray
import gc


class Map2D:
    def __init__(self, init_beam, mediums, h5_fname=None):
        self.init_beam = init_beam
        self.mediums = {m.__hash__(): m for m in mediums}

        for m in mediums:
            for o in m.objs:
                if hasattr(o, 'map'):
                    o.map = self  # store reference to self on objects that may need it

        self.t = init_beam.t

        # Calculate initial rays:
        # only simmetric
        self.rays = self.init_beam.rays  # store the beam rays in a list
        self.rays_h = [n.__hash__() for n in self.init_beam.rays]  # store the hash keys of the rays

        # self.t_solved = 0.

        if h5_fname is None:
            from tempfile import SpooledTemporaryFile
            h5_fname = SpooledTemporaryFile()

        # File is opened in init
        # Remember to run close eventually
        self.h5file = h5py.File(h5_fname, 'a')

    def calc_t(self, t=None, procs=None):
        t = self.t.max() if t is None else t

        # if t <= self.t_solved:
        #    return
        # import numpy as np
        # incs = np.linspace(self.t_solved, t, nincs+1)[1:]

        # o_rays = [r for r in self.rays]  # copy the original rays to propagate
        # Propagate all rays a time t
        if (procs is None) or (procs == 1):
            for i in tqdm(range(len(self.init_beam.rays))):
                ray = self.init_beam.rays[i]
                rays_r = ray.trace(t, self)  # returns hashes
                self.rays_h += rays_r
                gc.collect()

        else:
            raise NotImplementedError

            # from multiprocessing import Pool
            # p = Pool(procs)
            # r = []
            #
            # for ray in o_rays:
            #     args = [t, self.objs]
            #     r.append(p.apply_async(ray.trace, args))
            #
            # rays_r = [res.get() for res in r]
            # for res in rays_r:
            #     self.rays += res
        # self.t_solved = t

    def retrace_rays(self, length):
        """ Executes the retrace method on all rays stored in the map
        """
        for rh in self.rays_h:
            ray = self.get_ray(rh)
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
                       xmin=0, xmax=500., ymin=0., ymax=500., lvl_lim=7, lvl_N=22,
                       err_val=0.001
                       ):
        """

        :param ax: matplotlib ax object
        :param gridlen: lenght grid divisions
        :param t_ind: time index
        :param kind: ray kind: 'A0', 'S0', None
        :param xmin, xmax, ymin, ymax: Grid bounds
        :param lvl_lim:
        :param err_val:
        :return:
        """
        import numpy as np
        from scipy.fft import irfft

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
                err_val]

        rays_z = []
        z_val = np.zeros(X.T.shape)
        # for rh in self.rays_h:
        for i in tqdm(range(len(self.rays_h))):
            rh = self.rays_h[i]
            ray = self.get_ray(rh)
            if kind is not None:
                if kind != ray.kind:
                    continue
            # rays_z.append(split_ray(ray, *args))  #  for r in m.rays if r.kind != 'A0']
            z_val += split_ray(ray, *args)
        # z_val = sum(rays_z)

        if lvl_lim is None:
            lvl_lim = np.max(z_val)
        # levels = list(np.linspace(-lvl_lim, lvl_lim, 22))
        levels = list(lvl_lim*np.linspace(-1,1,lvl_N)**9)
        # tlvl = np.max(z_val / 2)
        # levels = set([-7, *[-tlvl + tlvl / 10 * n for n in range(20)], 7])
        # levels = list (levels)
        # levels.sort()
        # print(levels)
        # if len(levels)<
        ax.contourf(X, Y, z_val.T, vmin=-np.max(lvl_lim / 3), vmax=np.max(lvl_lim / 3), levels=levels)

        # x_axis = np.linspace(-gridsize/2, gridsize/2, 100)
        # y_axis = np.linspace(-gridsize/2, gridsize/2, 100)
        #
        # z_val = np.zeros([len(x_axis), len(y_axis)])
        #
        # for rh in self.rays_h:
        #     ray = self.get_ray(rh)
        #     for i, t in enumerate(ray.trace):
        #         zi = np.argmin(np.abs(x_axis - t[0]))
        #         zk = np.argmin(np.abs(y_axis - t[1]))
        #
        #         z_val[zi, zk] += ray.a[i] * irfft(ray.freq[i], n=len(ray.t))[t_ind]
        #
        # X, Y = np.meshgrid(x_axis, y_axis)
        # ax.contourf(X, Y, z_val)
        if self.mediums is not None:
            for m in self.mediums.values():
                for o in m.objs:
                    o.plot(ax)
        ax.set_aspect('equal')
        return ax, z_val, [X, Y]

    def to_vtk(self, filename, version='2.0', dtype='ASCII'):
        """ Writes paraview vtk file

        :param filename: VTK filename
        :param version: Paraview vtk version file, only 2.0 supported
        :return:
        """

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

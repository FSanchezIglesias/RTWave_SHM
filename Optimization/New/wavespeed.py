import os
import numpy as np
import h5py
from numba import jit
from scipy.interpolate import interp1d as scipy_interp1d # Keep as fallback/loader

# --- Numba Optimized Interpolation Kernels ---

@jit(nopython=True, cache=True)
def interp1d_numba(x, x_grid, y_grid):
    """
    Numba-compatible 1D interpolation using numpy.interp
    """
    # np.interp is supported in Numba nopython mode
    return np.interp(x, x_grid, y_grid)

@jit(nopython=True, cache=True)
def interp2d_regular_numba(point, x_grid, y_grid, data):
    """
    Fast Bilinear Interpolation for Regular Grids.
    point: [x, y] coordinates to query
    x_grid: 1D array of x coordinates (must be sorted/uniform)
    y_grid: 1D array of y coordinates (must be sorted/uniform)
    data: 2D array of values with shape (len(x_grid), len(y_grid))
    """
    x = point[0]
    y = point[1]
    
    # 1. Find indices
    # We assume uniform grids for maximum speed, or use searchsorted for non-uniform.
    # Using searchsorted is safer for general grids and still very fast in Numba.
    
    # Clip to bounds to avoid index errors (extrapolation = clamp)
    if x <= x_grid[0]: i = 0
    elif x >= x_grid[-1]: i = len(x_grid) - 2
    else: i = np.searchsorted(x_grid, x) - 1
    
    if y <= y_grid[0]: j = 0
    elif y >= y_grid[-1]: j = len(y_grid) - 2
    else: j = np.searchsorted(y_grid, y) - 1

    # Ensure i, j are within valid range for indexing
    if i < 0: i = 0
    if i >= len(x_grid) - 1: i = len(x_grid) - 2
    if j < 0: j = 0
    if j >= len(y_grid) - 1: j = len(y_grid) - 2

    # 2. Calculate local coordinates (0.0 to 1.0)
    x0 = x_grid[i]
    x1 = x_grid[i+1]
    y0 = y_grid[j]
    y1 = y_grid[j+1]
    
    # Avoid division by zero
    xd = x1 - x0
    yd = y1 - y0
    
    u = (x - x0) / xd if xd != 0 else 0.0
    v = (y - y0) / yd if yd != 0 else 0.0
    
    # 3. Bilinear Interpolation
    # f(u, v) = (1-u)(1-v)f00 + u(1-v)f10 + (1-u)v f01 + uv f11
    
    f00 = data[i, j]
    f10 = data[i+1, j]
    f01 = data[i, j+1]
    f11 = data[i+1, j+1]
    
    res = (1-u)*(1-v)*f00 + u*(1-v)*f10 + (1-u)*v*f01 + u*v*f11
    return res


@jit(nopython=True, cache=True)
def interp2d_batch_fixed_y(x_vals, y, x_grid, y_grid, data, result):
    """
    Batch bilinear interpolation for an array of x values at a fixed y.
    Computes the y-index and weights once, then loops over x values in compiled code.

    x_vals: 1D array of x coordinates to query
    y: scalar y coordinate (fixed for all queries)
    x_grid: 1D sorted array of x grid coordinates
    y_grid: 1D sorted array of y grid coordinates
    data: 2D array with shape (len(x_grid), len(y_grid))
    result: 1D output array, same length as x_vals (pre-allocated)
    """
    ny = len(y_grid)
    nx = len(x_grid)

    # --- y-index and weight (computed once) ---
    if y <= y_grid[0]:
        j = 0
    elif y >= y_grid[ny - 1]:
        j = ny - 2
    else:
        j = np.searchsorted(y_grid, y) - 1
    if j < 0:
        j = 0
    if j >= ny - 1:
        j = ny - 2

    y0 = y_grid[j]
    y1 = y_grid[j + 1]
    yd = y1 - y0
    v = (y - y0) / yd if yd != 0.0 else 0.0
    v1 = 1.0 - v

    # --- loop over x values ---
    for k in range(len(x_vals)):
        x = x_vals[k]

        if x <= x_grid[0]:
            i = 0
        elif x >= x_grid[nx - 1]:
            i = nx - 2
        else:
            i = np.searchsorted(x_grid, x) - 1
        if i < 0:
            i = 0
        if i >= nx - 1:
            i = nx - 2

        x0 = x_grid[i]
        x1 = x_grid[i + 1]
        xd = x1 - x0
        u = (x - x0) / xd if xd != 0.0 else 0.0

        result[k] = (1.0 - u) * (v1 * data[i, j] + v * data[i, j + 1]) \
                   + u * (v1 * data[i + 1, j] + v * data[i + 1, j + 1])

# --- Main Classes ---

class wavespeed:
    def __init__(self, foldername=None):

        self.S0 = None
        self.A0 = None
        self.Sh0 = None
        
        # Store raw data for JIT access
        self._S0_data = None 
        self._A0_data = None

        if foldername is not None:
            self.set_from_folder(foldername)

    def set_from_folder(self, foldername):
        for n in os.listdir(foldername):
            wname, t_file = n.split('.')

            if t_file == 'csv':
                # Returns (func, data_tuple)
                interp_func, data = self.read_csv(os.path.join(foldername, n))
                setattr(self, wname, interp_func)
                # Store raw data for potential future use
                if wname == 'S0': self._S0_data = data
                if wname == 'A0': self._A0_data = data

    @staticmethod
    def read_csv(filename, sep=','):
        point_arr = []
        with open(filename, 'r') as csv:
            for l in csv:
                point_arr.append([float(n) for n in l.split(sep)])
        point_arr = np.array(point_arr)
        
        # Sort by x just in case
        idx = np.argsort(point_arr[:, 0])
        x_grid = np.ascontiguousarray(point_arr[idx, 0])
        y_grid = np.ascontiguousarray(point_arr[idx, 1])

        # Create lambda that calls the JIT function
        # xt is expected to be [freq, angle] or just freq depending on usage.
        # Original code: interp(xt[0])
        interp_theta = lambda xt: interp1d_numba(xt[0], x_grid, y_grid)

        return interp_theta, (x_grid, y_grid)

    def calc_lamb(self, E, nu, rho, mod=8, f_int=10, maxf=10000):
        """
        :param E: Elastic modulus [Pa]
        :param nu: poisson modulus
        :param rho: density [kg/m3]
        :param mod: number of modes to calculate
        :param f_int: frequency interval [kHz]
        :param maxf: maximum frequency: [kHz]
        """
        from Lamb_disp.lamb_disp import disper

        # This returns m/s vs kHz
        f_raw, vps, vpa = disper(E, nu, rho, mod, f_int, maxf)
        
        # Convert to MHz for storage (matching original logic: f/1000)
        f_mhz = np.ascontiguousarray(f_raw / 1000.0)
        vps_data = np.ascontiguousarray(vps[:, 0])
        vpa_data = np.ascontiguousarray(vpa[:, 0])

        # Define optimized lambdas
        self.S0 = lambda xt: interp1d_numba(xt[0], f_mhz, vps_data)
        self.A0 = lambda xt: interp1d_numba(xt[0], f_mhz, vpa_data)
        
        # Store data for debug/direct access
        self._S0_data = (f_mhz, vps_data)
        self._A0_data = (f_mhz, vpa_data)


class wavespeed_composite:
    def __init__(self, filename):
        self._load_composite_hdf(filename)

    def _load_composite_hdf(self, filename):
        # We replace the read function to setup Numba interps
        with h5py.File(filename, 'r') as f:
            # Load and ensure C-contiguous for Numba
            self.f_S0 = np.ascontiguousarray(f['f_S0'])
            self.theta_S0 = np.ascontiguousarray(f['theta_S0'])
            self.R_S0 = np.ascontiguousarray(f['R_S0']).T # Transpose to match (x, y) indexing

            self.f_A0 = np.ascontiguousarray(f['f_A0'])
            self.theta_A0 = np.ascontiguousarray(f['theta_A0'])
            self.R_A0 = np.ascontiguousarray(f['R_A0']).T

        # Define wrappers using the JIT compiled kernel
        # We bind the arrays to the lambda using default args or closure
        
        # Optimization: Pre-bind these variables to avoid self lookup in the hot path
        f_s, t_s, r_s = self.f_S0, self.theta_S0, self.R_S0
        f_a, t_a, r_a = self.f_A0, self.theta_A0, self.R_A0

        self.S0 = lambda xt: interp2d_regular_numba(xt, f_s, t_s, r_s)
        self.A0 = lambda xt: interp2d_regular_numba(xt, f_a, t_a, r_a)

        # Batch lookup data — exposed for batch_speed calls
        self._grid = {
            'S0': (f_s, t_s, r_s),
            'A0': (f_a, t_a, r_a),
        }

    def batch_speed(self, kind, x_vals, y):
        """
        Batch interpolation: many x values (freq*th) at fixed y (theta).
        Returns result array of same length as x_vals.
        """
        f_grid, t_grid, data = self._grid[kind]
        result = np.empty(len(x_vals))
        interp2d_batch_fixed_y(x_vals, y, f_grid, t_grid, data, result)
        return result

    # Keep original method name if called externally, but it's now internal logic
    def read_composite_hdf(self, filename):
        self._load_composite_hdf(filename)
        # Return dummies or the lambdas if something expects return values
        return self.S0, self.A0
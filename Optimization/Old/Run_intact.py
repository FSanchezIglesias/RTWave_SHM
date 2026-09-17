# RUN dmg 8 mm center
import sys
import gc

import numpy as np
import h5py
from MUSE_dmg import gen_MUSE_intact

sys.path.append('./code')
from code.RayTracing.Ray import Beam_from_pzt

if __name__ == "__main__":
    t = np.linspace(0., 0.0002, 100000)
    nrays = 800
    f = 350.E+3
    plate_lenght = 726.
    xstep, ystep = 24., 24.
    xldmg, yldmg = 48., 48.
    hdf5_fname = 'Intact_{:.0f}k.hdf5'.format(f/1000)
    with h5py.File(hdf5_fname, 'w') as h5f:
        h5f['time'] = t
        h5f.attrs['Stacking'] = '(+45, -45, 90, 0)$'
        h5f.attrs['Material'] = 'AS4/8552'
        h5f.attrs['Plate dimensions'] = [plate_lenght, plate_lenght]

    # intact
    for source in range(8):
        m, pzts = gen_MUSE_intact()
    
        ibeam_i1 = Beam_from_pzt(nrays, pzts[source], power=nrays/2,
                                 f=f, npeaks=3, nfft=500, t=t)
        m.set_init_beam(ibeam_i1)
        m.calc_iter(3)
        m.calc_signal()
    
        key = 'Intact/PZT{}'.format(source+1)
        m.save_signals(hdf5_fname, key)
        with h5py.File(hdf5_fname, 'a') as h5f:
            h5f[key].attrs['dmg_type'] = 'Intact'
            # h5f[key].attrs['dmg_pos']=[np.nan, np.nan]
    
        m.close_h5()
        gc.collect()

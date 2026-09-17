# RUN dmg 8 mm center
import sys
import gc

import numpy as np
import h5py
from MUSE_dmg import gen_MUSE_dmg

# sys.path.append('/home/fer/Cosas/Doctorado/09_RayTracing_studies/code/')
from code.RayTracing.Ray import Beam_from_pzt

thdmg = 3.
rdmg = 0.1
bldmg = 0.05
rmdmg = 0.8

if __name__ == "__main__":
    xdmg = float(sys.argv[1])

    t = np.linspace(0., 0.0002, 100000)
    nrays = 800
    f = 350.E+3
    plate_lenght = 726.
    xstep, ystep = 24., 24.
    xldmg, yldmg = 48., 48.
    hdf5_fname = 'Tira02_{:.0f}k_x{:.2f}.hdf5'.format(f/1000,xdmg)
    with h5py.File(hdf5_fname, 'a') as h5f:
        try:
            h5f['time'] = t
            h5f.attrs['Stacking'] = '(+45, -45, 90, 0)$'
            h5f.attrs['Material'] = 'AS4/8552'
            h5f.attrs['Plate dimensions'] = [plate_lenght, plate_lenght]
        except OSError:
            pass

    # xdmg_i = np.linspace(xldmg/2, plate_lenght/2-xldmg/2, int(plate_lenght/(2*xstep)))
    ydmg_i = np.linspace(yldmg/2, plate_lenght/2-yldmg/2, int(plate_lenght/(2*ystep)))

    # dmg
    # for xdmg in xdmg_i:
    for ydmg in ydmg_i:
        for source in range(8):
            key = 'DGM48_{:.2f}_{:.2f}/PZT{}'.format(xdmg, ydmg, source+1)
            print(' --- Solving for: ' + key + ' --- ')
            with h5py.File(hdf5_fname, 'a') as h5f:
                if key in h5f:
                    print(key + ' already in ' + hdf5_fname)
                    continue
            #try:
            m, pzts = gen_MUSE_dmg(xdmg=xdmg, ydmg=ydmg, xldmg=xldmg, yldmg=yldmg, 
                                   thdmg=thdmg, rdmg=rdmg , bldmg=bldmg, rmdmg=rmdmg)
            ibeam_i1 = Beam_from_pzt(nrays, pzts[source], power=nrays/2,
                                     f=f, npeaks=3, nfft=500, t=t)
            m.set_init_beam(ibeam_i1)
            m.calc_iter(3)
            m.calc_signal()
        
            m.save_signals(hdf5_fname, key)
            with h5py.File(hdf5_fname, 'a') as h5f:
                h5f[key].attrs['dmg_type'] = '48 mm square damage'
                h5f[key].attrs['dmg_pos'] = [xdmg, ydmg]
                h5f[key].attrs['dmg_ch'] = [thdmg, rdmg, bldmg, rmdmg]
        
            m.close_h5()
            gc.collect()
            # except Exception as e:
            #     print(e)
            #     continue
        print(' - Done ydmg: {:.2f}'.format(ydmg))
    # # intact
    # for source in range(8):
    #     m, pzts = gen_MUSE_intact()
    # 
    #     ibeam_i1 = Beam_from_pzt(nrays, pzts[source], power=nrays/2,
    #                              f=f, npeaks=3, nfft=500, t=t)
    #     m.set_init_beam(ibeam_i1)
    #     m.calc_iter(3)
    #     m.calc_signal()
    # 
    #     key = 'Intact/PZT{}'.format(source+1)
    #     m.save_signals(hdf5_fname, key)
    #     with h5py.File(hdf5_fname, 'a') as h5f:
    #         h5f[key].attrs['dmg_type'] = 'Intact'
    #         # h5f[key].attrs['dmg_pos']=[np.nan, np.nan]
    # 
    #     m.close_h5()
    #     gc.collect()
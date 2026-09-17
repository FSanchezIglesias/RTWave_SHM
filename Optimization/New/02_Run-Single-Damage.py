# RUN dmg 12 mm square
import os, sys, gc, logging, h5py, time

import numpy as np
from MUSE_dmg import gen_MUSE_dmg
from RTWave_SHM.RayTracing.Ray import Beam_from_pzt

run_name = 'DMG12'

#Damage parameters
thdmg = 3.
rdmg = 0.1
bldmg = 0.05
rmdmg = 1

xdmg = 179.
ydmg = 145+4.


if __name__ == "__main__":

    start = time.time()

    logfile = os.path.join('logs','{}_row{:.2f}.log'.format(run_name, xdmg))
    # Configure logging
    logging.basicConfig(
        filename=logfile,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    t = np.linspace(0., 0.0002, 10_000)  # 50 MHz
    nrays = 2_001
    f = 350.E+3
    plate_lenght = 726.
    xstep, ystep = 12., 12.
    xldmg, yldmg = 12., 12.
    hdf5_fname = os.path.join('results',
                            '{}_{:.0f}k_x{:.2f}.hdf5'.format(run_name,f/1000,xdmg))
    with h5py.File(hdf5_fname, 'a') as h5f:
        try:
            h5f['time'] = t
            h5f.attrs['Stacking'] = '(+45, -45, 90, 0)$'
            h5f.attrs['Material'] = 'AS4/8552'
            h5f.attrs['Plate dimensions'] = [plate_lenght, plate_lenght]
        except OSError:
            print('Error A')
            pass

    print(f'\tydmg = {ydmg}')
    for source in range(8):
        print(f'\t\tPZT{source+1}')
        key = '{}_{:.2f}_{:.2f}/PZT{}'.format(run_name, xdmg, ydmg, source+1)
        logging.info(' --- Solving for: ' + key + ' --- ')
        with h5py.File(hdf5_fname, 'a') as h5f:
            if key in h5f:
                logging.warning(key + ' already in ' + hdf5_fname)
                continue
        #try:
        print(f'Source PZT{source+1}: Generating map.')
        m, pzts = gen_MUSE_dmg(
            xdmg=xdmg, ydmg=ydmg, xldmg=xldmg, yldmg=yldmg, 
            thdmg=thdmg, rdmg=rdmg , bldmg=bldmg, rmdmg=rmdmg
        )
        print(f'Source PZT{source+1}: Setting initial beam.')
        ibeam_i1 = Beam_from_pzt(
            nrays, pzts[source], power=2001/8,
            f=f, npeaks=3, nfft=500, t=t
        )
        print(f'Source PZT{source+1}: Generating beam with {nrays} rays.')
        m.set_init_beam(ibeam_i1)
        print(f'Source PZT{source+1}: Calculating travel times.')
        m.calc_t()
        print(f'Source PZT{source+1}: Calculating signals.')
        m.calc_signal()
    
        m.save_signals(hdf5_fname, key)
        with h5py.File(hdf5_fname, 'a') as h5f:
                if key in h5f:
                    # SUCCESS: The dataset exists
                    h5f[key].attrs['dmg_type'] = '12 mm square damage'
                    h5f[key].attrs['dmg_size'] = [xldmg, yldmg]
                    h5f[key].attrs['dmg_pos'] = [xdmg, ydmg]
                else:
                    # FAILURE: The dataset was not created
                    logging.error(f"FAILED to save dataset {key}. Signal might be empty.")
                    print(f"Skipping attributes for {key} (Dataset missing)")
    
        m.close_h5()
        gc.collect()
        # except Exception as e:
        #     print(e)
        #     continue

        end = time.time()
        print(end - start)

        break
    logging.info(' - Done ydmg: {:.2f}'.format(ydmg))

    
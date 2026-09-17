import numpy as np

import sys
# sys.path.append(r'C:\Users\Fer\Cosas\UPM\Doctorado\09_RayTracing_studies\code')
# sys.path.append(r'C:\Users\Fer\Cosas\UPM\Doctorado\09_RayTracing_studies')
sys.path.append('./code')

from code.Prop_2d import *
from Dispersion_curves.wavespeed import wavespeed_composite
#, wavespeed


# --- Constants ---   
ws = wavespeed_composite(r'Dispersion_curves/Composite/muse_stacking.hdf5')
th = 1.288  # mm

r_pzt = 4.
bl = 0.25

l = 726
#            l/4    l/5
pzt_pos = [[181.5, 145.2], 
           [181.5, 290.4],
           [181.5, 435.6],
           [181.5, 580.8],
           [544.5, 145.2],
           [544.5, 290.4],
           [544.5, 435.6],
           [544.5, 580.8]]

xdmg = l/2
xldmg = 8.

ydmg = l/2
yldmg = 8.

thdmg = 3.
rdmg = 0.1
bldmg = 0.05
rmdmg = 0.8

def gen_MUSE_dmg(ws=ws, r_pzt=r_pzt, bl=bl, l=l, th=th, pzt_pos=pzt_pos,
                 xdmg=xdmg, xldmg=xldmg, ydmg=ydmg, yldmg=yldmg,
                 wsdmg=ws, rdmg=rdmg, bldmg=bldmg, rmdmg=rmdmg, thdmg=thdmg):

    P01 = np.array([0., 0.])
    P02 = np.array([xdmg - xldmg/2, 0])
    P03 = np.array([xdmg + xldmg/2, 0])
    P04 = np.array([l, 0.])
    P05 = np.array([0., ydmg-yldmg/2])
    P06 = np.array([xdmg - xldmg/2, ydmg-yldmg/2])
    P07 = np.array([xdmg + xldmg/2, ydmg-yldmg/2])
    P08 = np.array([l, ydmg-yldmg/2])
    P09 = np.array([0., ydmg+yldmg/2])
    P10 = np.array([xdmg - xldmg/2, ydmg+yldmg/2])
    P11 = np.array([xdmg + xldmg/2, ydmg+yldmg/2])
    P12 = np.array([l, ydmg+yldmg/2])
    P13 = np.array([0., l])
    P14 = np.array([xdmg - xldmg/2, l])
    P15 = np.array([xdmg + xldmg/2, l])
    P16 = np.array([l, l])
    
    # Boundaries 
    S01 = Segment(P01, P02, boundary_losses=bl)
    S02 = Segment(P02, P03, boundary_losses=bl)
    S03 = Segment(P03, P04, boundary_losses=bl)
    S04 = Segment(P01, P05, boundary_losses=bl)
    S07 = Segment(P04, P08, boundary_losses=bl)
    S11 = Segment(P05, P09, boundary_losses=bl)
    S14 = Segment(P08, P12, boundary_losses=bl)
    S18 = Segment(P09, P13, boundary_losses=bl)
    S21 = Segment(P12, P16, boundary_losses=bl)
    S22 = Segment(P13, P14, boundary_losses=bl)
    S23 = Segment(P14, P15, boundary_losses=bl)
    S24 = Segment(P15, P16, boundary_losses=bl)
    
    # DMG walls
    S09 = Segment(P06, P07, boundary_losses=bldmg, ratio_rfl=rdmg, ratio_mode=rmdmg, color='blue')
    S12 = Segment(P06, P10, boundary_losses=bldmg, ratio_rfl=rdmg, ratio_mode=rmdmg, color='blue')
    S13 = Segment(P07, P11, boundary_losses=bldmg, ratio_rfl=rdmg, ratio_mode=rmdmg, color='blue')
    S16 = Segment(P10, P11, boundary_losses=bldmg, ratio_rfl=rdmg, ratio_mode=rmdmg, color='blue')
    
    # Inv walls
    S05 = Segment(P02, P06, boundary_losses=0., ratio_rfl=0., ratio_mode=1., color='white')
    S06 = Segment(P03, P07, boundary_losses=0., ratio_rfl=0., ratio_mode=1., color='white')
    S08 = Segment(P05, P06, boundary_losses=0., ratio_rfl=0., ratio_mode=1., color='white')
    S10 = Segment(P07, P08, boundary_losses=0., ratio_rfl=0., ratio_mode=1., color='white')
    S15 = Segment(P09, P10, boundary_losses=0., ratio_rfl=0., ratio_mode=1., color='white')
    S17 = Segment(P11, P12, boundary_losses=0., ratio_rfl=0., ratio_mode=1., color='white')
    S19 = Segment(P10, P14, boundary_losses=0., ratio_rfl=0., ratio_mode=1., color='white')
    S20 = Segment(P11, P15, boundary_losses=0., ratio_rfl=0., ratio_mode=1., color='white')
    
    # Medium initialization
    m1 = medium(ws, th, xi=1.e-3)
    m2 = medium(ws, th, xi=1.e-3)
    m3 = medium(ws, th, xi=1.e-3)
    m4 = medium(ws, th, xi=1.e-3)
    m5 = medium(ws, th, xi=1.e-3)
    m6 = medium(ws, th, xi=1.e-3)
    m7 = medium(ws, th, xi=1.e-3)
    m8 = medium(ws, th, xi=1.e-3)
    md = medium(wsdmg, thdmg, xi=5.e-3)
    
    m1.add_objs([S01, S04, S05, S08])
    m2.add_objs([S02, S05, S06, S09])
    m3.add_objs([S03, S06, S07, S10])
    m4.add_objs([S08, S11, S12, S15])
    m5.add_objs([S10, S13, S14, S17])
    m6.add_objs([S15, S18, S19, S22])
    m7.add_objs([S16, S19, S20, S23])
    m8.add_objs([S17, S20, S21, S24])
    md.add_objs([S09, S12, S13, S16])
    
    pzts = [Sensor('circ', [p, r_pzt], name='PZT{}'.format(i+1)) for i,p in enumerate(pzt_pos)]
    mediums = [m1, m2, m3, m4, m5, m6, m7, m8, md]

    m = Map2D(mediums=mediums, background=True)
    for k in pzts:
        try:
            m.add_sensor(k)
        except KeyError as e:
            print(e)
        
    return m, pzts    
#     
#     ibeam = Beam(nrays, [pzts[0].origin(),], power=nrays/2,
#                  medium=m1,
#                  f=f, npeaks=3, nfft=500, t=t)
#     
#     m = Map2D(ibeam, [th1,])
#     
#     return m
def gen_MUSE_intact(ws=ws, r_pzt=r_pzt, bl=bl, l=l, pzt_pos=pzt_pos):
    
    P01 = np.array([0., 0.])
    P02 = np.array([l, 0.])
    P03 = np.array([l, l])
    P04 = np.array([0., l])

    
    # Boundaries 
    S01 = Segment(P01, P02, boundary_losses=bl)
    S02 = Segment(P02, P03, boundary_losses=bl)
    S03 = Segment(P03, P04, boundary_losses=bl)
    S04 = Segment(P04, P01, boundary_losses=bl)
    
    # Medium initialization
    m1 = medium(ws, th, xi=1.e-3)
    
    m1.add_objs([S01, S02, S03, S04])
    
    pzts = [Sensor('circ', [p, r_pzt], name='PZT{}'.format(i+1)) for i,p in enumerate(pzt_pos)]
    mediums = [m1,]

    m = Map2D(mediums=mediums, background=True)
    for k in pzts:
        m.add_sensor(k)
        
    return m, pzts    
#

import sys
# sys.path.append('/home/c83330/Python/PyReader')  # Install the pyreader & Ditch package!!
pyreaderpath = '/home/fer/Cosas/Python/PyReader_cy/'
sys.path.append(pyreaderpath)

import re
import pandas as pd
# from matplotlib.ticker import FormatStrFormatter
import numpy as np
# import scipy.io as sio
# from scipy.signal import hilbert
from bdftools import look_for
import bdfcommons as bdfc
from read_output.read_abqrpt import read_abqrpt

# from Aux_functions import *
# from pyccx import *


def read_abq_CE2tests(feminp='/Data01/Cosas/Master_Space/02/CE2/01_FEM/00_RUN/CE2_v01_SteelCenter0.22kg0.5ms_200ms.inp',
                      femres='/Data01/Cosas/Master_Space/02/CE2/01_FEM/101_Reports/Active200kHz3p_PZT2_S_1lshell.rpt'):
    """ Reads results from MUSE panels abaqus run made in the CE2 format

    :param feminp: FEM model mesh
    :param femres: FEM results - rpt file
    :return: time, dict with things
    """

    FEModel = bdfc.FEModel()  # fem_file=feminp)
    c, FEModel = look_for(['elset', 'nset', 'element', 'node', 'include'], feminp, fe_model=FEModel)

    FEMh = read_abqrpt(femres)


    # sets defined as part:set <- new style for abaqus!
    pzt_sets = [('PZT1', 'panel'), ('PZT2', 'panel'), ('PZT3', 'panel'), ('PZT4', 'panel'),
                ('PZT5', 'panel'), ('PZT6', 'panel'), ('PZT7', 'panel'), ('PZT8', 'panel')]
    # pzt_sets = ['PZT1E', 'PZT2E', 'PZT3E', 'PZT4E', 'PZT5E', 'PZT6E', 'PZT7E', 'PZT8E']

    variables = ['S11', 'S12', 'S22', 'S33']
    dfs = {v: pd.DataFrame(index=FEMh.index) for v in variables}

    # Generate a new dataset for each var

    for col in FEMh.keys()[1:]:
        x_reg = re.search('S: *(\w+).*PI: *(\w+).*E: *(\d*)', col)
        if not x_reg:
            continue
        var_d = x_reg.group(1)
        if var_d in variables:
            part = x_reg.group(2)
            el = int(x_reg.group(3))
            area = FEModel.ELEMENTS[el].area()  # [part.lower()].
            dfs[var_d][el] = FEMh[col] * area

    pzt_s = {}
    for pzt, part in pzt_sets:
        pzt_signal = []
        A1 = sum([FEModel.ELEMENTS[el].area() for el in FEModel.CARD['elset'][
            pzt.lower()].members()]) * 1.E-3  # FEModel[part].elsets[pzt.lower()].members()])*1.E-03
        for mi in FEModel.CARD['elset'][pzt.lower()].members():  # FEModel[part].elsets[pzt.lower()].members():
            m = FEModel.ELEMENTS[mi]
            pzt_signal.append(pd.DataFrame(index=FEMh.index, columns=variables))
            for var_d in variables:
                # if var_d != 'S33':
                pzt_signal[-1][var_d] = dfs[var_d][m.ID] * m.area()
                # else:
                #    pzt_signal[-1][var_d] = 0.

        pzt_df = pd.DataFrame(index=FEMh.index, columns=variables)
        for var_d in variables:
            # if var_d != 'S33':
            pzt_df[var_d] = sum(
                [dfs[var_d][el] * FEModel.ELEMENTS[el].area() for el in FEModel.CARD['elset'][pzt.lower()].members()]) / A1
            # else:
            #    pzt2[var_d] = 0.*dfs['S11'][FEModel.elsets[pzt_set].members()[0].ID]
        pzt_s[pzt] = Vp_calc(pzt_df) / 2

    return FEMh['X'], pzt_s


def Vp_calc(df):
    from utils_rays.Constants_PZT import d, A, Cp

    # TODO: allow for either T or S...
    # Kind of important....
    # Retro compatibility...

    # Jaime
    # V = h/eps_0 * d31 * sum(s11+s22)Ai/At

    if isinstance(df, str):
        # return Vp_calc_old(df)
        raise TypeError('Old Vp_calc is lost in time, like tears in the rain...')

    # S = np.array([df['LE11'], df['LE22'], df['LE33'], df['LE12'], np.zeros(len(df.index)), np.zeros(len(df.index))])

    T = np.array([df['S11'], df['S22'], df['S33'], df['S12'], np.zeros(len(df.index)), np.zeros(len(df.index))])
    # T = np.matmul(c_E, S)

    # Voltage thingy?
    # TODO: Fix this shit
    D = np.matmul(d, T)
    Vp = D[2] * A / Cp
    return Vp

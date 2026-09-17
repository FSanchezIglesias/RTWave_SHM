import scipy.io as sio
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt


def Read_ScanGenie(fname):
    """ Read Scan Genie output files
        returns a dictionary with entries for each frequency
        
        r_SG[frequency][10*actuator+sensor] 
        ex.:    r_SG[250000.][25]['y'] == np.array([ input signal at sensor 2 at 250kHz])
                r_SG[250000.][25]['x'] == np.array([ signal from sensor 2 to 5 at 250kHz])
    """    
    # datos Christian panel plano: 'DATOS_PLACA/Christian_no_damage_1104.dat'
    Results_mat = sio.loadmat(fname)   
    
    # Transform the mat file into something more understandable
    r_SG = {}
    
    # Signal definition
    definition = Results_mat['setup'][0,0][6][0]
    for j in range(len(definition)):
        actuator = int(definition[j][0])
        sensor = int(definition[j][1])
        frequency = float(definition[j][4])

        #print(actuator, sensor, frequency)
        if frequency not in r_SG.keys():
            r_SG[frequency] = {}
        r_SG[frequency][10*actuator+sensor] = {
            'amplitude': float(definition[j][2]),
            'burst_type': str(definition[j][3]),
            'x': Results_mat['s'+str(j)].T[0],
            'y': Results_mat['a'+str(j)].T[0]
        } 
        
    return r_SG


def integral(x, y, t=None, t0=0, method='rect'):
    if t is None:
        t = x
        
    integ = [0.]
    
    f = interp1d(x, y, fill_value="extrapolate")
    
    t_m1 = t0
        
    for ti in t:
        if ti<=t0:
            i = 0.
        else:
            
            if method=='quad':
                i = integrate.quad(f, t0, ti)
            elif method=='rect':
                i = integ[-1] + 0.5*(ti-t_m1)*(f(ti)+f(t_m1))
        t_m1 = ti    
        integ.append(i)
        
    return np.array(integ[1:])


def fast_integral(x, y):
    integ = np.zeros(x.shape)

    for i, xi in enumerate(x[:-1]):
        integ[i+1] = integ[i] + 0.5 * (x[i+1] - xi) * (y[i+1] + y[i])

    return integ


# --- SCAN GENIE DATA PROCESSING FUNCTIONS ---
def get_sgth(sgdata, f, path, norm=False, sampling_rate=48.E+06):
    if not norm:
        norm_arr = 1.
    elif 'S' in str(norm):
        # symmetric wave -> maximum after 4.00E-05 s 
        ind_sym = int(2e-5 * sampling_rate)
        norm_arr = np.max(np.abs(sgdata[f][path]['x'][ind_sym:]))
        
    elif 'C' in str(norm):
        # crosstalk -> assumes always maximum of signal
        norm_arr = np.max(np.abs(sgdata[f][path]['x']))
        
    else: # Default / inp ignal
        norm_arr = np.max(np.abs(sgdata[f][path]['y']))

    return sgdata[f][path]['x']/norm_arr


def plot_sgdata(sgdata, ax, f, path, norm=False, sgname=' ', cmap=plt.cm.gnuplot2, linestyle='-',
                fs= 32000, sampling_rate=48.E+06):
    
    # colors = cmap(np.linspace(0,1, len(path_l)+len(to_plot)))
    
    label = sgname + ', f = {:.1f} kHz, path: '.format(f/1000) + str(path)
    
    th = get_sgth(sgdata, f, path, norm)
    
    tmax = fs / sampling_rate
    t_test = np.linspace(0, tmax, fs)
    
    if isinstance(ax, np.ndarray):
        ax[0].plot(t_test, th, label=label, linestyle=linestyle)
        ax[1].plot(t_test, np.abs(hilbert(th)), label=label, linestyle=linestyle)
        
        if ax.shape[0]>2:
            ax[2].plot(t_test, integral(t_test, np.abs(th)), label=label, linestyle=linestyle)
            
            
def energy_fromth(sgdata, f, path, norm=False,
                  tini=0, tend=None,
                  fs= 32000, sampling_rate=48.E+06):
    tmax = fs / sampling_rate 
    t_test = np.linspace(0, tmax, fs)
    
    i_ini = int(tini*sampling_rate)
    
    th = get_sgth(sgdata, f, path, norm)
    th = th[i_ini:]
    t_test = t_test[i_ini:]
    
    if tend is not None:
        i_end = int(tend*sampling_rate)
        th = th[:i_end]
        t_test = t_test[:i_end]
    
    # return integral(x=t_test, y=th)
    return integrate.simps(np.abs(th), t_test)
    

def plot_sensors(sgdata, ax=None,
                 tini=2.e-5, tend=6.5e-04, vlim = 1.6,
                 fftlim=1500.e+3, figsize=(16,22), nfigs=4
                ):
    """
    :param sgdata: dictionary {'label':[x_values, y_values], ...}
    :param ax: matplotlib axes object
    :return: fig, ax
    """
    
    if ax is None:
        fig, ax = plt.subplots(nfigs,1, figsize=figsize)
    else:
        fig = None
    
    for l, data in sgdata.items():
        ax[0].plot(data[0], data[1], label=l)
        ax[1].plot(data[0], np.abs(hilbert(data[1])), label=l)
        
        i_integral = int(tini*len(data[0])/data[0][-1])
        if nfigs > 2:
            ax[2].plot(data[0][i_integral:], fast_integral(data[0][i_integral:], np.abs(data[1][i_integral:])), label=l)
        
        i_end = int(tend*len(data[0])/data[0][-1])
        if nfigs > 3:
            ax[3].plot(*fft_plot(data[0][i_integral:i_end], data[1][i_integral:i_end]), label=l)
        
    ax[0].axhline(0, color='black')
    ax[0].legend()
    ax[0].grid()
    
    ax[0].set_xlim(tini, tend)
    ax[0].set_title('Signal')
    # ax[0].set_title('Frequency: {:.0f} kHz'.format(f/1000))
    ax[0].set_xlabel('time [ms]')
    
    ax[1].axhline(0, color='black')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_xlim(tini, tend)
    ax[1].set_title('Hilbert transform')
    ax[1].set_xlabel('time [ms]')
    
    if nfigs > 2:
        ax[2].axhline(0, color='black')
        ax[2].legend()
        ax[2].grid()
        ax[2].set_xlim(tini, tend)
        ax[2].set_title('Integral - accumulated')
        ax[2].set_xlabel('time [ms]')
    
    
    # ax[2].axhline(0, color='black')
    if nfigs > 3:
        ax[3].legend()
        ax[3].grid()
        ax[3].set_xlim(0., fftlim)
        ax[3].set_title('FFT')
        ax[3].set_xlabel('Frequency [Hz]')
    
    if vlim is not None:
        ax[0].set_ylim(-vlim, vlim) 
        ax[1].set_ylim(0., vlim) 
    return fig, ax


def fft_plot(x, y):
    
    N = len(x)
    dt = (max(x)-min(x))/N
    
    yf = np.fft.fft(y)
    xf = np.linspace(0.0, 0.5/dt, N//2)

    return xf, 2.0/N * np.abs(yf[:int(N//2)])
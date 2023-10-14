import numpy as np
from scipy.signal.windows import hamming, hann


class Signal:
    def __init__(self, npeaks, f=0., a=1.):
        self.npeaks = npeaks
        self.a = a
        self.f = f

    def signal(self, ti, f=None, a=None):
        f = self.f if f is None else f
        a = self.a if a is None else a

        signal = np.sin(2 * np.pi * f * ti)
        std = 1 / (2 * np.pi * f) * self.npeaks
        signal = a * np.exp(-(ti - 4 * std) ** 2 / (2 * std ** 2)) * signal

        return signal


def burst_normal(t, a, f, fd, phase=0, delay=8.e-07):
    signal = np.sin(2 * np.pi * f * (t-delay) + phase)
    # std = 1/(2 * np.pi * fd)
    mu = 1/(2 * fd)
    # signal = a * np.exp(-(t - mu - delay) ** 2 / (2 * std ** 2)) * signal
    signal = a * np.exp(-0.5*((t-mu-delay)*2*np.pi*fd)**2)*signal

    return signal


def burst_hamming(t, a, f, fd, phase=0., delay=8.e-07):
    """ Sin burst windowed with a hamming
    T window = 1/fd

    BURST3 -> fd = f/3
    BURST5 -> fd = f/5
    """
    signal = np.zeros(t.shape)

    T = 1/fd
    t_window = (delay < t) & (t < T+delay)
    sint = np.sin(2*np.pi*f*(t-delay)+phase)

    n = sum(t_window)
    signal[t_window] = hamming(n) * sint[t_window] * a
    return signal


def burst_hann(t, a, f, fd, phase=0., delay=8.e-07):
    """ Sin burst windowed with a hamming
    T window = 1/fd

    BURST3 -> fd = f/3
    BURST5 -> fd = f/5
    """
    signal = np.zeros(t.shape)

    T = 1/fd
    t_window = (delay < t) & (t < T+delay)
    sint = np.sin(2*np.pi*f*(t-delay)+phase)

    n = sum(t_window)
    signal[t_window] = hann(n) * sint[t_window] * a
    return signal

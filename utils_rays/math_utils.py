import numpy as np


# Integral function...
# kind of slow I guess
def integral(x, y, t=None, t0=0, method='rect'):
    from scipy.integrate import quad
    from scipy.interpolate import interp1d

    if t is None:
        t = x

    integ = [0.]

    f = interp1d(x, y, fill_value="extrapolate")

    t_m1 = t0

    for ti in t:
        if ti <= t0:
            i = 0.
        else:

            if method == 'quad':
                i = quad(f, t0, ti)
            elif method == 'rect':
                i = integ[-1] + 0.5 * (ti - t_m1) * (f(ti) + f(t_m1))
            else:
                raise TypeError('Unknown integration schema: {}'.format(method))
        t_m1 = ti
        integ.append(i)

    return np.array(integ[1:])


def fast_integral(x, y):
    integ = np.zeros(x.shape)

    for i, xi in enumerate(x[:-1]):
        integ[i+1] = integ[i] + 0.5 * (x[i+1] - xi) * (y[i+1] + y[i])

    return integ


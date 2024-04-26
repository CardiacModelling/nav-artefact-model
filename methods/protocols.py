#
# Pre-defined voltage-clamp protocols
#
import myokit
import pints
import numpy as np

from . import data2 as data


def load_pname(pname, dname):
    if 'NaIV_' in pname:
        dt = 0.04  # ms; NOTE: This has to match the data
        if dname in data.batch1:
            print(f'{dname} is in data.batch1')
            discard_start = 9
            remove = 25 + discard_start
            step_duration = 40 - remove  # ms
            discard = remove + 2000  # ms
            v_steps = data._naiv(dname)
            protocol = load('protocols/ina-steps.txt')
        elif dname in data.batch2:
            print(f'{dname} is in data.batch2')
            discard_start = 9
            remove = 25 + discard_start
            step_duration = 40 - remove  # ms
            discard = remove + 200  # ms
            v_steps = data._naiv(dname)
            protocol = load('protocols/ina-steps-2.txt')
        elif dname in data.batch3 + data.batch4:
            print(f'{dname} is in data.batch3 + data.batch4')
            discard_start = 8
            # remove = 25 + discard_start
            remove = 8 + discard_start
            step_duration = 39 - remove  # ms
            discard = remove + 2000  # ms
            v_steps = data._naiv(dname)
            protocol = load('protocols/ina-steps-3.txt')
        else:
            raise ValueError(f'{dname} is not given in methods.data.batch.')

    elif 'NaInact_' in pname:
        dt = 0.04  # ms; NOTE: This has to match the data
        if dname in data.batch3 + data.batch4:
            print(f'{dname} is in data.batch3 + data.batch4')
            discard_start = 1008
            remove = 31 + discard_start
            step_duration = 1049 - remove  # ms
            discard = remove + 991  # ms
            v_steps = data._nainact(dname)
            protocol = load('protocols/inact-steps-3.txt')

        else:
            raise ValueError(f'{dname} is not given in methods.data.batch.')

    return dt, discard_start, remove, step_duration, discard, v_steps, protocol


def load(filename):
    """
    Creates a :class:`myokit.Protocol` from a step file using
    :meth:`load_steps(filename)` and :meth:`from_steps()`.
    """
    return from_steps(load_steps(filename))


def load_steps(filename):
    """ Loads a list of voltage protocol steps from ``filename``. """
    return np.loadtxt(filename).flatten()


def from_steps(p):
    """
    Takes a list ``[step_1_voltage, step_1_duration, step_2_voltage, ...]`` and
    returns a :clas:`myokit.Protocol`.
    """
    protocol = myokit.Protocol()
    for i in range(0, len(p), 2):
        protocol.add_step(p[i], p[i + 1])
    return protocol


def generate_filter(p, dt=1, duration=5):
    """
    Generates a capacitance filter based on a :class:`myokit.Protocol`: each
    ``duration`` time units after every step will be filtered out.
    """
    times = np.arange(0, duration, self.dt)
    mask = np.ones(times.shape, dtype=bool)
    for step in p:
        mask *= times < p.start
        mask *= times >= p.start + p.duration
    return mask


def fold(times, data, fold, discard=0, discard_start=0):
    """
    Fold the data into a dict of {0:[data_fold_0], 1:[data_fold_1], ...},
    return folded_time, folded_data.

    times: time series of the data.
    data: the data to be folded.
    fold: duration for each fold (same unit as times).
    discard: duration for which to be discarded after each fold.
    discard_start: if non-zero, discard the beginning of the data by duration
                   of discard_start.
    """
    dt = times[1] - times[0]
    ti = times[0]
    if discard_start:
        ti += discard_start
    tj = ti + fold
    tf = times[-1]
    out = dict()
    i = 0
    while tj <= tf + dt:
        out[i] = data[((times >= ti) & (times < tj))]
        ti = tj + discard
        tj = ti + fold
        i += 1
    t = np.arange(0, fold, dt)
    return t, out


def mask(times, fold, discard=0, discard_start=0):
    """
    Fold the data into a dict of {0:[data_fold_0], 1:[data_fold_1], ...},
    return folded_time, folded_data.

    times: time series of the data.
    fold: duration for each fold (same unit as times).
    discard: duration for which to be discarded after each fold.
    discard_start: if non-zero, discard the beginning of the data by duration
                   of discard_start.
    """
    dt = times[1] - times[0]
    ti = times[0]
    if discard_start:
        ti += discard_start
    tj = ti + fold
    tf = times[-1]
    m = np.full(len(times), False)
    while tj <= tf + dt:
        m[((times >= ti) & (times < tj))] = True
        ti = tj + discard
        tj = ti + fold
    return m


def naiv_iv(times, data_folded, dname, is_data=False, vs=None):
    """
    Return (current, voltage) for the I-V relationship of protocol NaIV.
    """
    if vs is None:
        vs = data._naiv(dname)
    if dname in data.batch3 + data.batch4:
        wins = [9.2, 13]
    else:
        wins = [10.2, 14]
    m = ((times > wins[0]) & (times < wins[1]))
    current = []
    for i, v in enumerate(vs):
        k = v if is_data else i
        c = np.argmax(np.abs(data_folded[k][m]))
        current.append(np.array(data_folded[k][m])[c])
    return current, vs


def naiv_inact(times, data_folded, dname, is_data=False):
    """
    Return (current, voltage) for the I-V relationship of protocol NaIV.
    """
    vs = data._nainact(dname)
    if dname in data.batch3 + data.batch4:
        wins = [1009.2, 1013]
    else:
        raise ValueError(f'No inact data for {dname}')
    m = ((times > wins[0]) & (times < wins[1]))
    current = []
    for i, v in enumerate(vs):
        k = v if is_data else i
        c = np.argmax(np.abs(data_folded[k][m]))
        current.append(np.array(data_folded[k][m])[c])
    return current, vs


class ActivationIV(object):
    def __init__(self, times, step_duration, discard, discard_start, t_start,
                 dname):
        self._times = times
        self._step_duration = step_duration
        self._discard = discard
        self._discard_start = discard_start
        self._t_start = t_start
        self._dname = dname

    def __call__(self, data, is_data=False):
        t, c = fold(self._times, data, self._step_duration,
                    discard=self._discard, discard_start=self._discard_start)

        #import matplotlib.pyplot as plt
        #for i in range(len(c)):
        #    plt.plot(t, c[i])
        #plt.show()

        i_iv, v_iv = naiv_iv(t + self._t_start, c, self._dname, is_data=False)
        return i_iv


from scipy.optimize import curve_fit
class ActivationIVBiomarkers(object):
    def __init__(self, times, step_duration, discard, discard_start, t_start,
                 dname):
        self._times = times
        self._step_duration = step_duration
        self._discard = discard
        self._discard_start = discard_start
        self._t_start = t_start
        self._dname = dname

        self._dt = self._times[1] - self._times[0]
        self._tfold = np.arange(0, self._step_duration, self._dt)
        self._tfold += self._t_start
        self._mask = self._get_mask()
        self._vs = data._naiv(self._dname)

        if self._dname in data.batch3 + data.batch4:
            self._step_time = 9
        else:
            self._step_time = 10

    def __call__(self, data, is_data=False):
        t, c = fold(self._times, data, self._step_duration,
                    discard=self._discard, discard_start=self._discard_start)

        #import matplotlib.pyplot as plt
        #for i in range(len(c)):
        #    plt.plot(t, c[i])
        #plt.show()

        idx = self._peak_indices(c)
        i_iv = self._iv(c, idx)
        tau_f, tau_s = self._deactivation_tau(c, idx)
        t_peak = self._time_to_peak(c, idx)

        if is_data:
            nan = (np.abs(i_iv) < 500)  # pA
            tau_f[nan] = np.NaN
            tau_s[nan] = np.NaN
            t_peak[nan] = np.NaN

        return np.array([i_iv, tau_f, tau_s, t_peak]).flatten()

    def iv(self, data):
        t, c = fold(self._times, data, self._step_duration,
                    discard=self._discard, discard_start=self._discard_start)
        idx = self._peak_indices(c)
        i_iv = self._iv(c, idx)
        return i_iv

    def voltage(self):
        return self._vs

    def _get_mask(self):
        if self._dname in data.batch3 + data.batch4:
            wins = [9.05, 13]
        else:
            wins = [10.05, 14]
        m = ((self._tfold > wins[0]) & (self._tfold < wins[1]))
        return m

    def _peak_indices(self, data_folded):
        idx = []
        for i, v in enumerate(self._vs):
            k = i
            idx.append(np.argmax(np.abs(data_folded[k][self._mask])))
        return idx

    def _iv(self, data_folded, idx):
        current = []
        for i, v in enumerate(self._vs):
            k = i
            current.append(np.array(data_folded[k][self._mask])[idx[i]])
        return np.array(current)

    def _time_to_peak(self, data_folded, idx):
        t = []
        for i, v in enumerate(self._vs):
            k = i
            t.append(self._tfold[self._mask][idx[i]] - self._step_time)
        return np.array(t)

    def _activation_tau(self, data_folded, idx):
        # NOTE: maybe not this but use time to peak.
        tau = []
        for i, v in enumerate(self._vs):
            k = i
            y = np.array(data_folded[k][self._mask])[:idx[i]]
            x = self._tfold[self._mask][:idx[i]] - self._step_time
            p0 = (-1, 0.5)
            po, qo = curve_fit(lambda t,a,b: a*np.exp(t/b), x, y, p0=p0)
            tau.append(po[1])
        return np.array(tau)

    def _deactivation_tau(self, data_folded, idx):
        tau_f = []
        tau_s = []
        peaks = self._iv(data_folded, idx)
        #import matplotlib.pyplot as plt
        for i, v in enumerate(self._vs):
            k = i
            y = np.array(data_folded[k][self._mask])[idx[i]:]
            x = self._tfold[self._mask][idx[i]:]
            #x -= self._step_time
            x -= x[0]
            #plt.plot(x, y)
            p0 = (-np.abs(peaks[i])*1.5, 0.2, -np.abs(peaks[i])*0.15, 2.)
            try:
                po, qo = curve_fit(
                    lambda t,a,b,c,d: a*np.exp(-t/b) + c*np.exp(-t/d),
                    x, y, p0=p0)
                tau_f.append(po[1] if po[1] < 1 else 1e7)
                tau_s.append(po[3] if po[3] < 100 else 1e7)
                #plt.plot(x, po[0]*np.exp(-x/po[1]) + po[2]*np.exp(-x/po[3]), '--')
            except:
                tau_f.append(1e7)
                tau_s.append(1e7)
        #plt.show()
        return np.array(tau_f), np.array(tau_s)


class ModifiedMeanSquaredError(pints.ProblemErrorMeasure):
    r"""
     Calculates a sum of squares error:

    .. math::
        f = \sum _i^n (y_i - x_i) ^ 2,

    where :math:`y` is the data, :math:`x` the model output, both of which are
    modified through a function :math:`h`, and :math:`n` is the total number of
    data points after applying :math:`h`.

    Extends :class:`ErrorMeasure`.

    Parameters
    ----------
    problem
        A :class:`pints.SingleOutputProblem` or
        :class:`pints.MultiOutputProblem`.
    function
        A function that modifies the data and the model output (with an
        optional arg `is_data` to specify data input for any special
        consideration).
    """
    def __init__(self, problem, weights=None, function=None):
        super(ModifiedMeanSquaredError, self).__init__(problem)

        if function is None:
            function = lambda x,is_data=False: x
        self._h = function
        values = []
        for i in range(self._n_outputs):
            if self._n_outputs > 1:
                values.append(self._h(self._values[:, i], is_data=True))
            else:
                values.append(self._h(self._values[:], is_data=True))
        self._values = np.array(values).T
        self._ninv = 1.0 / np.product(self._values.shape)

        if weights is None:
            weights = np.ones(self._values.shape)
        self._weights = np.asarray(weights)**2  # weight within squares

    def __call__(self, x):
        o = np.ones(self._values.shape) * float('inf')
        y = self._problem.evaluate(x)
        for i in range(self._n_outputs):
            try:
                if self._n_outputs > 1:
                    o[:, i] = self._h(y[:, i])
                else:
                    o[:, i] = self._h(y[:])
            except: # TODO check
                pass
        return np.sum(((np.nansum(((o - self._values)**2) * self._weights,
                              axis=0)) * self._ninv),
                      axis=0)

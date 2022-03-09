#
# Pre-defined voltage-clamp protocols
#
import myokit
import numpy as np


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


def fold(times, data, fold, discard=0, discard_start=False):
    """
    Fold the data into a dict of {0:[data_fold_0], 1:[data_fold_1], ...},
    return folded_time, folded_data.

    times: time series of the data.
    data: the data to be folded.
    fold: duration for each fold (same unit as times).
    discard: duration for which to be discarded after each fold.
    discard_start: if True, discard the beginning of the data by duration of
                   discard.
    """
    dt = times[1] - times[0]
    ti = times[0]
    if discard_start:
        ti += discard
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


def mask(times, fold, discard=0, discard_start=False):
    """
    Fold the data into a dict of {0:[data_fold_0], 1:[data_fold_1], ...},
    return folded_time, folded_data.

    times: time series of the data.
    fold: duration for each fold (same unit as times).
    discard: duration for which to be discarded after each fold.
    discard_start: if True, discard the beginning of the data by duration of
                   discard.
    """
    dt = times[1] - times[0]
    ti = times[0]
    if discard_start:
        ti += discard
    tj = ti + fold
    tf = times[-1]
    m = np.full(len(times), False)
    while tj <= tf + dt:
        m[((times >= ti) & (times < tj))] = True
        ti = tj + discard
        tj = ti + fold
    return m

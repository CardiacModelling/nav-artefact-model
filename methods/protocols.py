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
    raise NotImplementedError

#
# Data loading methods
#
import myokit
import numpy as np

from . import DIR_METHOD


def load_named(dname, pname=None, cell=None, model=None, parameters=None):
    """
    Generates or loads a named data set.

    A valid name is either:

    - a data set name ``dname`` and a protocol name ``pname`` with ``cell``,
    - a data set name ``dname`` and a ``model`` with a set of ``parameters``,

    where ``dname`` is one of ``['synth1', 'real1']``.
    """
    if dname == 'syn':
        raise NotImplementedError
        tr, vr, cr = fake_data(model, parameters, sigma=0.1, seed=1)  # TODO
    elif dname == 'real':
        cells = {# ID, file_name
            '0':'220128_006_ch2_csv',
            '1':'220210_003_ch3_csv',
        }
        if 'NaIV' in pname:
            tr, vr, cr = load_inaiv(
                f'{DIR_METHOD}/../data/{cells[cell]}/{pname}')
        else:
            raise ValueError('Unknow `pname`, expecting `NaIV`.')
    else:
        raise ValueError('Unknonw `dname`, expecting `syn` or `real`.')
    return tr, vr, cr


def load_inaiv(path):
    """
    Loads an "Alex" file: CSV with current (A/F).

    Returns a tuple ``(t, v, c)`` where ``t`` is time in ms, ``v`` is voltage
    in mV, and ``c`` is current in A/F.
    """
    v_steps = [-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40]  # mV
    v_hold = -100.  # mV
    t_hold = 10.    # ms
    t_step = 20.    # ms
    dt = 0.04       # ms (25 kHz)
    times = np.arange(0, t_hold + t_step + t_hold, dt)
    voltage = dict()
    data = dict()
    for v in v_steps:
        vt = v_hold * np.ones(int(t_hold / dt))
        vt = np.append(vt, float(v) * np.ones(int(t_step / dt)))
        vt = np.append(vt, float(v_hold) * np.ones(int(t_hold / dt)))
        voltage[v] = vt
        data[v] = np.loadtxt(f'{path}/step_{v}.csv', delimiter=',')
    return times, voltage, data


def fake_data(model, parameters, sigma, seed=None):
    """
    Generates fake "Alex" style data, by running simulations with the given
    ``model`` and ``parameters`` and adding noise with ``sigma``.

    If a ``seed`` is passed in a new random generator will be created with this
    seed, and used to generate the added noise.
    """
    t = model.times()
    v = model.voltage()
    c = model.simulate(parameters)

    sigma = 0.1
    if seed is not None:
        # Create new random generator, leave the shared one unaltered.
        r = np.random.default_rng(seed=seed)
        c += r.normal(scale=sigma, size=c.shape)
    else:
        c += np.random.normal(scale=sigma, size=c.shape)

    return t, v, c


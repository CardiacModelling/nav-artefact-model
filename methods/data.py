#
# Data loading methods
#
import myokit
import numpy as np

from . import DIR_METHOD

_cnames = {# ID, file_name
    'cell0':'220128_006_ch2_csv',
    'cell1':'220210_003_ch3_csv',
}

_naiv_steps = [-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40]  # mV

DIR_DATA = f'{DIR_METHOD}/../data'


def data_sets(include_synth=True):
    """ Returns a list of available data sets. """
    names = list(_cnames.keys())
    if include_synth:
        names = ['syn'] + names
    return names


def load_named(dname, pname=None, model=None, parameters=None):
    """
    Generates or loads a named data set.

    A valid name is either:
    
    - a data set name ``dname`` and a protocol name ``pname``,
    - a data set name ``dname`` and a ``model`` with a set of ``parameters``,
    
    where ``dname`` is one of ``['syn', 'cell1', 'cell2', ...]`` (for a full
    list see :meth:`data_sets()`.
    """
    if dname == 'syn':
        raise NotImplementedError
        return fake_data(model, parameters, sigma=0.1, seed=1)  # TODO

    try:
        cname = _cnames[dname]
    except KeyError:
        raise ValueError(f'Unknown data set {dname}')
    if 'NaIV' in pname:
        return load_naiv(f'{DIR_DATA}/{cname}/{pname}')
    else:
        raise ValueError(f'Unknown protocol {pname}')


def load_naiv(path):
    """
    Loads an "Alex" file: CSV with current (nA).

    Returns a tuple ``(t, v, c)`` where ``t`` is time in ms, ``v`` is voltage
    in mV, and ``c`` is current in pA.
    """
    v_steps = list(_naiv_steps)
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
        data[v] *= 1e3  # nA -> pA
    return times, voltage, data


def get_naiv_alphas(path):
    import re
    if 'NaIVCP' in path:
        alpha = np.float(re.findall('NaIVCP(\d*)', path)[0])
        alpha_r = alpha_p = alpha / 100.
    elif ('NaIVC' in path) and not ('NaIVCP' in path):
        alpha = np.float(re.findall('NaIVC(\d*)', path)[0])
        alpha_r = alpha / 100.
        alpha_p = 0
    elif ('NaIVP' in path) and not ('NaIVCP' in path):
        alpha = np.float(re.findall('NaIVC(\d*)', path)[0])
        alpha_r = 0
        alpha_p = alpha / 100.
    else:
        raise ValueError('Unknown alpha for {path}')
    return alpha_r, alpha_p


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


def load_info(cname):
    """
    Return (cm, rs) of the data `cell` in pF and GOhm.
    """
    import pandas as pd
    info_file = 'info.csv'
    info = pd.read_csv(f'{DIR_DATA}/{info_file}', index_col=0, header=[0])
    cell = _cnames[cname]
    return info.loc[cell]['cm'], info.loc[cell]['rs'] * 1e-3  # M -> G

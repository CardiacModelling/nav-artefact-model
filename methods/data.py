#
# Data loading methods
#
import myokit
import numpy as np

from . import DIR_METHOD, models

_cnames = {# ID, file_name
    'cell1':'batch1/220128_006_ch2_csv',
    'cell2':'batch1/220210_003_ch3_csv',
}

_pnames = [f'NaIVCP{i}' for i in np.arange(0, 90, 10)] \
        + [f'NaIVC{i}' for i in np.arange(0, 90, 10)] \
        + [f'NaIVP{i}' for i in np.arange(0, 90, 10)]

_naiv_steps = [-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40]  # mV

DIR_DATA = f'{DIR_METHOD}/../data/nav'


def data_sets(include_synth=True):
    """ Returns a list of available data sets. """
    names = list(_cnames.keys())
    if include_synth:
        names = ['syn'] + names
    return names


def protocol_sets():
    """ Returns a list of available protocol sets. """
    return list(_pnames)


def load_named(dname, pname=None, model=None, parameters=None, shift=False):
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
        return load_naiv(f'{DIR_DATA}/{cname}/{pname}', shift=shift)
    else:
        raise ValueError(f'Unknown protocol {pname}')


def load_naiv(path, leakcorrect=False, shift=False):
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
        if leakcorrect:
            # Use the first 5ms holding at -100mV to leak correct the data.
            # Assuming E_leak = 0 mV.
            I_5ms = np.mean(data[v][:int(5. / dt)])
            g_leak_est = I_5ms / v_hold
            E_leak_est = 0  # assumption only
            data[v] -= g_leak_est * (voltage[v] - E_leak_est)
        if shift:
            # Use the first 5ms holding at -100mV to shift the data to 0pA.
            I_5ms = np.mean(data[v][:int(5. / dt)])
            data[v] -= I_5ms
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
    assert(alpha_r < 1. and alpha_r >= 0.)
    assert(alpha_p < 1. and alpha_p >= 0.)
    return alpha_r, alpha_p


def get_naiv_base(path):
    if 'NaIVCP' in path:
        return 'NaIVCP'
    elif ('NaIVC' in path) and not ('NaIVCP' in path):
        return 'NaIVC'
    elif ('NaIVP' in path) and not ('NaIVCP' in path):
        return 'NaIVP'
    else:
        raise ValueError('Unknown alpha for {path}')


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


def setup_model_vc(dname, model):
    """
    Prepare VCModel object for voltage clamp parameters with a given dname.
    """
    is_full = model.vc_level() == models.VC_FULL
    is_min = model.vc_level() == models.VC_MIN
    if is_full or is_min:
        if 'syn' not in dname:
            cm, rs = load_info(dname)
            g_leak = 1. # 1 GOhm seal; TODO Update this by experimental data.
        else:
            raise NotImplementedError
        model.set_artefact_parameters({
            'voltage_clamp.Cm_est':cm,
            'voltage_clamp.R_series_est':rs,
            'voltage_clamp.g_leak_est':g_leak,
            # Values to infer by scaling
            'cell.Cm':cm,
            'voltage_clamp.R_series':rs,
            'voltage_clamp.g_leak':g_leak,
            'voltage_clamp.E_leak':0,  # Imperfect seal leak
        })

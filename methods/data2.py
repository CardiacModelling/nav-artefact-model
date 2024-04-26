#
# Data loading methods
#
import myokit
import numpy as np

from . import DIR_METHOD, models

_cnames = {# ID, file_name
    'cell1':'220128_006_ch2_csv',
    'cell2':'220210_003_ch3_csv',
    'cell3':'220314_001_ch1_csv',
    'cell4':'220314_003_ch2_csv',
    'cell5':'220314_004_ch2_csv',
    'cell6':'nav_iv_and_inact/medium_res_data/220502_008_ch2',
    'cell7':'nav_iv_and_inact/medium_res_data/220502_009_ch3',
    'cell8':'nav_iv_and_inact/medium_res_data/220502_011_ch1',
    'cell9':'nav_iv_and_inact/medium_res_data/220502_011_ch3',
    'cell10':'nav_iv_and_inact/medium_res_data/220503_001_ch1',
    'cell11':'nav_iv_and_inact/medium_res_data/220503_001_ch2',
    'cell12':'nav_iv_and_inact/medium_res_data/220503_002_ch1',
    'cell13':'nav_iv_and_inact/medium_res_data/220503_002_ch2',
    'cell14':'nav_iv_and_inact/medium_res_data/220503_003_ch3',
    'cell15':'nav_iv_and_inact/medium_res_data/220503_004_ch1',
    'cell16':'nav_iv_and_inact/medium_res_data/220503_004_ch4',
    'cell17':'nav_iv_and_inact/medium_res_data/220503_005_ch1',
    'cell18':'nav_iv_and_inact/medium_res_data/220503_005_ch2',
    'cell19':'nav_iv_and_inact/medium_res_data/220503_005_ch3',
    'cell20':'nav_iv_and_inact/medium_res_data/220503_005_ch4',
}

_pnames = [f'NaIV_{j}C_{i}CP' for i in np.arange(0, 90, 20) for j in [25, 35]]
_pnames2 = [f'NaInact_{j}C_{i}CP' for i in np.arange(0, 90, 20) for j in [35]]

batch1 = ['cell1', 'cell2']
batch2 = ['cell3', 'cell4', 'cell5']
batch3 = ['cell6', 'cell7', 'cell8', 'cell9', 'cell10', 'cell11', 'cell12',
          'cell13', 'cell14', 'cell15', 'cell16', 'cell17', 'cell18', 'cell19',
          'cell20']


def _naiv(cell):
    if cell in batch1:
        raise ValueError('batch1 data should use data.py instead of data2.py')
    elif cell in batch2:
        _naiv_steps = np.arange(-80, 40, 5)  # mV; cells 3-5
    elif cell in batch3:
        _naiv_steps = np.arange(-80, 70, 10)  # mV; cells 6+
    else:
        raise ValueError(f'No _naiv protocol defined for {cell}')
    return _naiv_steps


def _nainact(cell):
    if cell in batch1:
        raise ValueError(f'No _nainact measurements for {cell}')
    elif cell in batch2:
        raise ValueError(f'No _nainact measurements for {cell}')
    elif cell in batch3:
        _nainact_steps = np.arange(-120, -20, 10)  # mV; cells 6+
    return _nainact_steps


DIR_DATA = f'{DIR_METHOD}/../data'


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
        return load_naiv(f'{DIR_DATA}/{cname}/{pname}', dname, shift=shift)
    elif 'NaInact' in pname:
        return load_nainact(f'{DIR_DATA}/{cname}/{pname}', dname, shift=shift)
    else:
        raise ValueError(f'Unknown protocol {pname}')


def load_naiv(path, dname, leakcorrect=False, shift=False):
    """
    Loads an "Alex" file: CSV with each column (voltage header) for
    current (nA).

    Returns a tuple ``(t, v, c)`` where ``t`` is time in ms, ``v`` is voltage
    in mV, and ``c`` is current in pA.
    """
    import pandas as pd
    _naiv_steps = _naiv(dname)  # Get cell/batch data protocol info
    v_hold = -100.  # mV
    v_steps = list(_naiv_steps)
    if dname in batch2:
        t_hold1 = 10.    # ms; cells 1-5
    if dname in batch3:
        t_hold1 = 9.    # ms; cells 6+ TODO: check
    t_step = 20.    # ms
    t_hold2 = 10.    # ms
    dt = 0.04       # ms (25 kHz)
    times = np.arange(0, t_hold1 + t_step + t_hold2, dt)
    voltage = dict()
    data = dict()
    df = pd.read_csv(f'{path}.csv', header=0)
    for v in v_steps:
        vt = v_hold * np.ones(int(t_hold1 / dt))
        vt = np.append(vt, float(v) * np.ones(int(t_step / dt)))
        vt = np.append(vt, float(v_hold) * np.ones(int(t_hold2 / dt)))
        voltage[v] = vt
        data[v] = df['{:.1f}'.format(v)]
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
    del(df)
    return times, voltage, data


def load_nainact(path, dname, leakcorrect=False, shift=False):
    """
    Loads an "Alex" file: CSV with each column (voltage header) for
    current (nA).

    Returns a tuple ``(t, v, c)`` where ``t`` is time in ms, ``v`` is voltage
    in mV, and ``c`` is current in pA.
    """
    raise NotImplementedError
    import pandas as pd
    _nainact_steps = _nainact(dname)  # Get cell/batch data protocol info
    v_hold = -100.  # mV
    v_steps = list(_nainact_steps)
    v_test = -20.   # mV
    t_hold = 9.     # ms
    t_step = 1000.  # ms
    t_test = 40.    # ms
    dt = 0.04       # ms (25 kHz)
    times = np.arange(0, t_hold + t_step + t_test, dt)
    voltage = dict()
    data = dict()
    df = pd.read_csv(f'{path}.csv', header=0)
    for v in v_steps:
        vt = v_hold * np.ones(int(t_hold / dt))
        vt = np.append(vt, float(v) * np.ones(int(t_step / dt)))
        vt = np.append(vt, float(v_test) * np.ones(int(t_test / dt)))
        voltage[v] = vt
        data[v] = df['{:.1f}'.format(v)]
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
    del(df)
    return times, voltage, data


def get_naiv_alphas(path):
    """ Return (alpha_R, alpha_P). """
    import re
    if 'NaIV_' in path:
        alpha = float(re.findall('NaIV_\d\dC_(\d*)CP', path)[0])
        alpha_r = alpha_p = alpha / 100.
    else:
        raise ValueError('Unknown alpha for {path}')
    assert(alpha_r < 1. and alpha_r >= 0.)
    assert(alpha_p < 1. and alpha_p >= 0.)
    return alpha_r, alpha_p


def get_naiv_temperature(path):
    """ Return temperature in K. """
    import re
    if 'NaIV_' in path:
        temperature = int(re.findall('NaIV_(\d*)C_.*CP', path)[0])
    else:
        raise ValueError('Unknown temperature for {path}')
    assert(temperature in [25, 35])
    return 273.15 + float(temperature)


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
    # TODO: Use meta data for Cm and Rs?
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
        if 'syn' in dname:
            raise NotImplementedError
        else:
            cm, rs = load_info(dname)
            g_leak = 1. # 1 GOhm seal; TODO Update this by experimental data.
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

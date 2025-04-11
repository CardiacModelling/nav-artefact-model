#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
from scipy.signal import find_peaks
import pandas as pd
import myokit

'''
This script generates simulated data for the INa model using Latin Hypercube
sampling to demonstrate averaging IV curve issues.
'''

MULTIPROCESSING = False
CASE = 3

# Cases 1 to 5; Rs lower and upper bounds in MOhm
CASES = {1: (0.5, 2),
         2: (2, 5),
         3: (5, 10)}

def lhs_array(min_val, max_val, n, log=False):
    sampler = qmc.LatinHypercube(1)
    sample = sampler.random(n)
    if log:
        dat = 10**qmc.scale(sample, np.log10(min_val), np.log10(max_val))
    else:
        dat = qmc.scale(sample, min_val, max_val)
    return dat.flatten()


def simulate_model(mod, proto, with_hold=True, sample_freq=0.00004):
    if mod.time_unit().multiplier() == .001:
        scale = 1000
    else:
        scale = 1
    p = mod.get('engine.pace')
    p.set_binding(None)
    v_cmd = mod.get('voltageclamp.Vc')
    v_cmd.set_rhs(0)
    v_cmd.set_binding('pace') # Bind to the pacing mechanism
    # Run for 20 s before running the VC protocol
    if with_hold:
        holding_proto = myokit.Protocol()
        holding_proto.add_step(-.080*scale, 30*scale)
        sim = myokit.Simulation(mod, holding_proto)
        t_max = holding_proto.characteristic_time()
        sim.run(t_max)
        mod.set_state(sim.state())
    t_max = proto.characteristic_time()
    times = np.arange(0, t_max, sample_freq*scale)
    sim = myokit.Simulation(mod, proto)
    dat = sim.run(t_max, log_times=times)
    return dat, times


def get_iv_data(mod, dat, times):
    iv_dat = {}
    cm = mod['voltageclamp']['cm_est'].value()
    i_out = [v/cm for v in dat['voltageclamp.Iout']]
    v = np.array(dat['voltageclamp.Vc'])
    step_idxs = np.where(np.diff(v) > .005)[0]
    v_steps = v[step_idxs + 10]
    iv_dat['Voltage'] = v_steps
    currs = []
    for idx in step_idxs:
        temp_currs = i_out[(idx+3):(idx+103)]
        x = find_peaks(-np.array(temp_currs), distance=5, width=4)
        if len(x[0]) < 1:
            currs.append(np.min(temp_currs))
        else:
            currs.append(temp_currs[x[0][0]])
    iv_dat['Current'] = currs
    return iv_dat


def mod_sim(param_vals):
    proto = myokit.Protocol()
    for v in range(-90, 50, 5):
        proto.add_step(-100, 2000)
        proto.add_step(v, 20)
    gna, rs, cm = param_vals
    comp = .8
    mod = myokit.load_model('gray_na_lei.mmt')
    mod['INa']['g_Na_scale'].set_rhs(gna)
    mod['voltageclamp']['rseries'].set_rhs(rs)
    mod['voltageclamp']['rseries_est'].set_rhs(rs)
    mod['voltageclamp']['cm_est'].set_rhs(cm)
    mod['model_parameters']['Cm'].set_rhs(cm)
    mod['voltageclamp']['alpha_c'].set_rhs(comp)
    mod['voltageclamp']['alpha_p'].set_rhs(comp)
    dat, times = simulate_model(mod, proto)
    iv_dat = get_iv_data(mod, dat, times)
    return [iv_dat, param_vals]


def generate_dat(num_mods=5):
    gna_vals = lhs_array(.2, 5, n=num_mods, log=True)
    rslb, rsub = CASES[CASE] # MOhm
    rs_vals = lhs_array(rslb*1E-3, rsub*1E-3, n=num_mods) # GOhm
    cm_vals = lhs_array(50, 150, n=num_mods)
    vals = np.array([gna_vals, rs_vals, cm_vals]).transpose()
    if MULTIPROCESSING:
        from multiprocessing import Pool
        with Pool() as p:
            dat = p.map(mod_sim, vals)
    else:
        dat = []
        for i, p in enumerate(vals):
            print(i, p)
            dat.append(mod_sim(p))
    all_currents = []
    all_meta = []
    for curr_mod in dat:
        all_currents.append(curr_mod[0]['Current'])
        all_meta.append(curr_mod[1])
    all_sim_dat = pd.DataFrame(all_currents, columns=dat[0][0]['Voltage'])
    mod_meta = pd.DataFrame(all_meta, columns=['G_Na', 'Rs', 'Cm'])
    all_sim_dat.to_csv(f'./data/case-{CASE}/all_iv.csv', index=False)
    mod_meta.to_csv(f'./data/case-{CASE}/all_param.csv', index=False)


def main():
    generate_dat(50)


if __name__ == '__main__':
    main()

# %%
import numpy as np
import matplotlib.pyplot as plt
import myokit
import pints
import pandas as pd
from scipy.signal import find_peaks

import warnings
warnings.filterwarnings('ignore')

# %%
v_steps = np.arange(-90, 50, 5)

def setup_simulation(sample_freq=0.00004):
    mod = myokit.load_model('gray-2020-ina.mmt')
    if mod.time_unit().multiplier() == .001:
        scale = 1000
    else:
        scale = 1
    # Run for 20 s before running the VC protocol
    holding_proto = myokit.Protocol()
    holding_proto.add_step(-.080*scale, 30*scale)
    holding_t_max = holding_proto.characteristic_time()
    holding_sim = myokit.Simulation(mod, holding_proto)
    # Run the VC protocol
    proto = myokit.Protocol()
    for v in v_steps:
        proto.add_step(-100, 2000)
        proto.add_step(v, 20)
    t_max = proto.characteristic_time()
    times = np.arange(0, t_max, sample_freq*scale)
    sim = myokit.Simulation(mod, proto)
    return holding_sim, sim, holding_t_max, t_max, times


def simulate_model_full(param, holding_sim, sim, holding_t_max, t_max, times):
    gna, p1, p2, p3, p4, taum, tauh, deltah = param
    holding_sim.reset()
    holding_sim.set_constant('ina.p1', p1)
    holding_sim.set_constant('ina.p2', p2)
    holding_sim.set_constant('ina.p3', p3)
    holding_sim.set_constant('ina.p4', p4)
    holding_sim.set_constant('ina.taum', taum)
    holding_sim.set_constant('ina.tauh', tauh)
    holding_sim.set_constant('ina.deltah', deltah)
    holding_sim.set_constant('ina.g_Na', gna)
    holding_sim.run(holding_t_max)
    sim.reset()
    sim.set_state(holding_sim.state())
    sim.set_constant('ina.p1', p1)
    sim.set_constant('ina.p2', p2)
    sim.set_constant('ina.p3', p3)
    sim.set_constant('ina.p4', p4)
    sim.set_constant('ina.taum', taum)
    sim.set_constant('ina.tauh', tauh)
    sim.set_constant('ina.deltah', deltah)
    sim.set_constant('ina.g_Na', gna)
    dat = sim.run(t_max, log_times=times)
    return dat


def simulate_model(param, holding_sim, sim, holding_t_max, t_max, times):
    gna, p1 = param
    holding_sim.reset()
    # Em = Em_original + p1; Eh = Eh_original + p3
    # Want both Em and Eh to change TOGETHER such that the time constants
    # are not affected while changing the IV curve.
    holding_sim.set_constant('ina.p1', p1)
    holding_sim.set_constant('ina.p3', p1)
    holding_sim.set_constant('ina.g_Na', gna)
    holding_sim.run(holding_t_max)
    sim.reset()
    sim.set_state(holding_sim.state())
    sim.set_constant('ina.p1', p1)
    sim.set_constant('ina.p3', p1)
    sim.set_constant('ina.g_Na', gna)
    dat = sim.run(t_max, log_times=times)
    return dat


def get_iv_data(dat):
    iv_dat = {}
    i_out = [v for v in dat['ina.i_Na']]
    v = np.array(dat['engine.pace'])
    step_idxs = np.where(np.diff(v) > .005)[0]
    v_steps = v[step_idxs + 10]
    iv_dat['Voltage'] = v_steps
    #print(v_steps)
    currs = []
    for idx in step_idxs:
        #temp_currs = i_out[(idx+3):(idx+103)]
        temp_currs = i_out[(idx):(idx+103)]
        #plt.plot(temp_currs)
        x = find_peaks(-np.array(temp_currs)) #distance=5, width=4)
        if len(x[0]) < 1:
            currs.append(np.min(temp_currs))
        else:
            currs.append(temp_currs[x[0][0]])
        #plt.axhline(currs[-1])
    iv_dat['Current'] = currs
    return iv_dat


# %%
#default_param_full = [20, 0, 0, 0, 0, 0.12, 6.45, 0.755]
default_param = [20, 0]

class pints_model(pints.ForwardModel):
    def __init__(self):
        self.holding_sim, self.sim, self.holding_t_max, self.t_max, self.times = setup_simulation()
        self.n_params = 2

    def n_parameters(self):
        return self.n_params

    def simulate(self, parameters, times=None):
        # parameters_full = gna, p1, p2, p3, p4, taum, tauh, deltah
        # parameters = gna, p1=p3
        dat = simulate_model(parameters, self.holding_sim, self.sim,
                             self.holding_t_max, self.t_max, self.times)
        iv_dat = get_iv_data(dat)
        return iv_dat['Current']

    def voltage(self):
        return v_steps

# gNa
transform_1 = pints.LogTransformation(n_parameters=1)
## p1, p2, p3, p4
#transform_2 = pints.IdentityTransformation(n_parameters=4)
## taum, tauh, deltah
#transform_3 = pints.LogTransformation(n_parameters=3)
# p1
transform_2 = pints.IdentityTransformation(n_parameters=1)

# The full transformation: [r, K] -> [r, log(K)]
transformation = pints.ComposedTransformation(transform_1, transform_2)#, transform_3)

model = pints_model()

# %%
'''
data = np.loadtxt('./data/baseline.csv')

# %%
problem = pints.SingleOutputProblem(model, model.voltage()+100, data)
error = pints.SumOfSquaresError(problem)
opt = pints.OptimisationController(
    error,
    default_param,
    transformation=transformation,
    method=pints.CMAES,
    )
opt.set_parallel(False)
opt.set_max_unchanged_iterations(iterations=100, threshold=1)
found_parameters, found_value = opt.run()

# %%
np.savetxt(f'./data/gray2020-optimal_params-baseline.csv', found_parameters)
plt.plot(model.voltage(), data)
plt.plot(model.voltage(), model.simulate(found_parameters))
plt.savefig('./data/gray2020-optimal_params-baseline.png')
plt.close()
'''
# %%
CASES = [1, 2, 3]

for CASE in CASES:
    all_iv = pd.read_csv(f'./data/case-{CASE}/all_iv.csv').T
    np.random.seed(0)
    choice = np.random.choice(len(all_iv.T), 25, replace=False)
    d = []
    for iv in all_iv.values[:, choice].T: d.append(iv)
    data = np.mean(d, axis=0)

    problem = pints.SingleOutputProblem(model, model.voltage()+100, data)
    error = pints.SumOfSquaresError(problem)
    opt = pints.OptimisationController(
        error,
        default_param,
        transformation=transformation,
        method=pints.CMAES,
        )
    opt.set_parallel(False)
    opt.set_max_unchanged_iterations(iterations=100, threshold=1)
    found_parameters, found_value = opt.run()
    np.savetxt(f'./data/gray2020-optimal_params-case-{CASE}.csv', found_parameters)

    plt.plot(model.voltage(), data)
    plt.plot(model.voltage(), model.simulate(found_parameters))
    plt.savefig(f'./data/gray2020-optimal_params-case-{CASE}.png')
    plt.close()

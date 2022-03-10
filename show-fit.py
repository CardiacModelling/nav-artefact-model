#!/usr/bin/env python3
#
# Fit a model to a current trace.
#
#!/usr/bin/env python3
import os

import numpy as np
import pints

from methods import data, utils, models, protocols
from methods import results, run, t_hold, v_hold

# Get model name, vc level, protocol name, data name, and experiment name
mname, level, pnames, dname, ename = utils.cmd('Perform a fit')

# Show user what's happening
print('=' * 79)
print(' '.join([f'Run {run}',
                mname,
                f'vc_level {level}',
                ' '.join(pnames),
                dname,
                f't_hold {t_hold}']))
print('=' * 79)

# Load protocol
dt = 0.04  # ms; NOTE: This has to match the data
step_duration = 40  # ms
discard = 2000  # ms
v_steps = data._naiv_steps
protocol = protocols.load('protocols/ina-steps.txt')

# Load alpha values
alphas = []
for pname in pnames:
    alphas.append(
        data.get_naiv_alphas(pname)
    )

# Create simple VC model
print('Initialising model...')
model = models.VCModel(
    models.mmt(mname),
    fit_kinetics=True,
    fit_artefacts=True,
    vc_level=level,
    alphas=alphas if level != models.VC_IDEAL else None,
    E_leak=True,
)
model.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold)
mask = protocols.mask(model.times(), step_duration, discard=discard)
model.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold, mask=mask)

# Create parameter vector
n_parameters = model.n_parameters()
parameters_true = np.ones(n_parameters)

# Generate or load data
crs = []
for pname in pnames:
    tr, vr_d, cr_d = data.load_named(dname, pname, model, parameters_true)
    cr = []
    for v in v_steps:
        cr = np.append(cr, cr_d[v])
    crs.append(cr)
crs = np.asarray(crs).T  # (n_times, n_outputs)
tr = np.arange(0, dt * len(cr), dt)

# Set voltage clamp setting
data.setup_model_vc(dname, model)

# Show current best results
path = os.path.join(results, ename)
parameters, info = utils.load(
    os.path.join(path, 'result.txt'), n_parameters=n_parameters)
utils.show_summary(parameters, info)

import matplotlib.pyplot as plt
plt.plot(tr, crs)
plt.plot(tr, model.simulate(parameters[0]))
plt.show()


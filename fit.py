#!/usr/bin/env python3
#
# Fit a model to a current trace.
#
#!/usr/bin/env python3
import os

import numpy as np
import pints

from methods import data, fitio, models, protocols
from methods import results, run, t_hold, v_hold

# Set random seed
np.random.seed(0)

# Get model name, protocol name, data name, and experiment name
mname, pnames, dname, ename = fitio.cmd('Perform a fit')

# Show user what's happening
print('=' * 79)
print(' '.join([f'Run {run}', mname, ' '.join(pnames), dname, f't_hold {t_hold}']))
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
    vc_level=models.VC_FULL,
    alphas=alphas,
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
if 'syn' not in dname:
    cm, rs = data.load_info(dname)
else:
    raise NotImplementedError
model.set_artefact_parameters({
    'voltage_clamp.Cm_est':cm,
    'voltage_clamp.R_series_est':rs,
    'voltage_clamp.g_leak_est':0,
    'voltage_clamp.E_leak':v_hold,
})

# Create boundaries
class LogRectBounds(pints.RectangularBoundaries):
    """
    Rect boundaries, but samples in 2-log space.
    """
    def sample(self, n=1):
        """ See :meth:`pints.Boundaries.sample()`. """
        lo = np.log2(self._lower)
        hi = np.log2(self._upper)
        xs = np.random.uniform(lo, hi, size=(n, self._n_parameters))
        return 2**xs


b = 1000
boundaries = LogRectBounds(
    np.ones(n_parameters) / b, np.ones(n_parameters) * b)

# Create score function
if len(pnames) > 1:
    problem = pints.MultiOutputProblem(model, tr, crs)
else:
    problem = pints.SingleOutputProblem(model, tr, cr)
error = pints.MeanSquaredError(problem)

# Create transformation for scaling factor parameters
transformation = pints.LogTransformation(model.n_parameters())

# Try fitting
path = os.path.join(results, ename)
fitio.fit(path, error, boundaries, transformation, 10, 50)

# Show current best results
parameters, info = fitio.load(
    os.path.join(path, 'result.txt'), n_parameters=n_parameters)
fitio.show_summary(parameters, info)


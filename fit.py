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

# Get model name, protocol name, data name, and experiment name
mname, pname, dname, ename = fitio.cmd('Perform a fit')

# Show user what's happening
print('=' * 79)
print(' '.join([f'Run {run}', mname, pname, dname, f't_hold {t_hold}']))
print('=' * 79)

# Load protocol
protocol = protocols.load('protocols/ina-steps.txt')
dt = 0.04

# Create simple VC model
print('Initialising model...')
model = models.VCModel(models.mmt(mname), True, True, models.VC_FULL)
model.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold)

# Create parameter vector
n_parameters = model.n_parameters()
parameters_true = np.ones(n_parameters)

# Generate or load data
tr, vr_d, cr_d = data.load_named(dname, pname, model, parameters_true)
cr = []
for v in data._naiv_steps:
    cr = np.append(cr, cr_d[v])
tr = np.arange(0, dt * len(cr), dt)

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
problem = pints.SingleOutputProblem(model, tr, cr)
error = pints.MeanSquaredError(problem)

# Try fitting
path = os.path.join(results, ename)
fitio.fit(path, error, boundaries, None, 10, 50)

# Show current best results
parameters, info = fitio.load(
    os.path.join(path, 'result.txt'), n_parameters=n_parameters)
fitio.show_summary(parameters, info)


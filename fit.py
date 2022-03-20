#!/usr/bin/env python3
#
# Fit a model to a current trace.
#
#!/usr/bin/env python3
import os

import numpy as np
import pints

#from methods import data as data
from methods import data2 as data
from methods import utils, models, protocols
from methods import results, run, t_hold, v_hold

# Get model name, vc level, protocol name, data name, and experiment name
mname, level, pnames, dname, ename = utils.cmd('Perform a fit')
GUESS = not True

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
discard = 0 # 2000  # ms
v_steps = data._naiv_steps
#protocol = protocols.load('protocols/ina-steps.txt')
protocol = protocols.load('protocols/ina-steps-2-no-holding.txt')

# Load alpha values
alphas = []
for pname in pnames:
    alphas.append(
        data.get_naiv_alphas(pname)
    )

# Load temperature values
temperatures = []
for pname in pnames:
    try:
        temperatures.append(
            data.get_naiv_temperature(pname)
        )
    except AttributeError:
        temperatures.append(None)
if not (temperatures.count(temperatures[0]) == len(temperatures)):
    raise ValueError('Expect all protocols have the same temperature.')
temperature = temperatures[0]

# Create simple VC model
print('Initialising model...')
model = models.VCModel(
    models.mmt(mname),
    fit_kinetics=True,
    fit_artefacts=True,
    vc_level=level,
    alphas=alphas if level != models.VC_IDEAL else None,
    E_leak=True,
    temperature=temperature,
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
    tr, vr_d, cr_d = data.load_named(dname, pname, model, parameters_true, shift=True)
    cr = []
    for v in v_steps:
        cr = np.append(cr, cr_d[v])
    crs.append(cr)
crs = np.asarray(crs).T  # (n_times, n_outputs)
tr = np.arange(0, dt * len(cr), dt)

# Set voltage clamp setting
data.setup_model_vc(dname, model)

# Initial guess of the parameters
if GUESS:
    guess, _ = utils.load(os.path.join(
        results,
        #f'results-test-{mname}-vc1-{dname}-NaIVCP80',
        f'results-test-{mname}-vc1-{dname}-NaIV_35C_80CP',
        'result.txt'), n_parameters=model.n_kinetics_parameters()+1)
    guess = np.append(guess[0], np.ones(model.n_parameters() - len(guess[0])))
    print('Initial guess:', guess)
else:
    guess = None

# Create boundaries and transformation
class LogRectBounds(pints.RectangularBoundaries):
    """
    Rect boundaries, but samples around 1.
    """
    def sample(self, n=1):
        """ See :meth:`pints.Boundaries.sample()`. """
        xs = 1.
        xs += np.random.normal(0, 0.1, size=(n, self._n_parameters))
        return xs

b = 1e5
def setup_bound_and_transform(model, b):
    """
    Setup and return boundaries and transformation.
    """
    # Check if we have voltage offset in the parameters
    p = model.fit_parameter_names()
    n_parameters = model.n_parameters()
    assert(len(p) == n_parameters)
    if 'voltage_clamp.V_offset_eff' in p:
        i = p.index('voltage_clamp.V_offset_eff')
        print(f'Parameter {i} is voltage offset, with no transformation.')
        l = np.ones(i) / b
        l = np.append(l, -30)
        l = np.append(l, np.ones(n_parameters - i - 1) / b)
        u = np.ones(i) * b
        u = np.append(u, 30)
        u = np.append(u, np.ones(n_parameters - i - 1) * b)
    else:
        l = np.ones(n_parameters) / b
        u = np.ones(n_parameters) * b
    # Create boundaries
    boundaries = LogRectBounds(l, u)
    # Create transformation for bounded parameters
    transformation = pints.RectangularBoundariesTransformation(boundaries)
    return boundaries, transformation

boundaries, transformation = setup_bound_and_transform(model, b)

# Create score function
if len(pnames) > 1:
    problem = pints.MultiOutputProblem(model, tr, crs)
else:
    problem = pints.SingleOutputProblem(model, tr, cr)
error = pints.MeanSquaredError(problem)


# Try fitting
path = os.path.join(results, ename)
utils.fit(path, error, boundaries, transformation, 10, 50, guess=guess)

# Show current best results
parameters, info = utils.load(
    os.path.join(path, 'result.txt'), n_parameters=n_parameters)
utils.show_summary(parameters, info)


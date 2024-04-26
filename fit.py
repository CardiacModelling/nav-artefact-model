#!/usr/bin/env python3
#
# Fit a model to a current trace.
#
#!/usr/bin/env python3
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pints

#from methods import data as data
from methods import data2 as data
from methods import utils, models, protocols
from methods import results, run, t_hold, v_hold

# Get model name, vc level, protocol name, data name, and experiment name
mname, level, pnames, dname, ename = utils.cmd('Perform a fit')
FIT_KINETICS = True
GUESS = not True
ENFORCE_SIMILAR = not True

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
if dname in data.batch1:
    print(f'{dname} is in data.batch1')
    discard_start = 9
    remove = 25 + discard_start
    step_duration = 40 - remove  # ms
    discard = remove + 2000  # ms
    v_steps = data._naiv(dname)
    protocol = protocols.load('protocols/ina-steps.txt')
elif dname in data.batch2:
    print(f'{dname} is in data.batch2')
    discard_start = 9
    remove = 25 + discard_start
    step_duration = 40 - remove  # ms
    discard = remove + 200  # ms
    v_steps = data._naiv(dname)
    protocol = protocols.load('protocols/ina-steps-2.txt')
elif dname in data.batch3:
    print(f'{dname} is in data.batch3')
    discard_start = 8
    remove = 25 + discard_start
    step_duration = 39 - remove  # ms
    discard = remove + 2000  # ms
    v_steps = data._naiv(dname)
    protocol = protocols.load('protocols/ina-steps-3.txt')
else:
    raise ValueError(f'{dname} is not given in methods.data.batch.')

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
    fit_kinetics=FIT_KINETICS,
    fit_artefacts=True,
    vc_level=level,
    alphas=alphas if level != models.VC_IDEAL else None,
    E_leak=True,
    temperature=temperature,
)
model.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold)
mask = protocols.mask(model.times(), step_duration, discard=discard,
                      discard_start=discard_start)
model.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold, mask=mask)

# Create parameter vector
n_parameters = model.n_parameters()
parameters_true = np.ones(n_parameters)

# Generate or load data
crs = []
for pname in pnames:
    tr, vr_d, cr_d = data.load_named(dname, pname, model, parameters_true,
                                     shift=True)
    cr = []
    for v in v_steps:
        cr = np.append(cr, cr_d[v])
    crs.append(cr)
crs = np.asarray(crs).T  # (n_times, n_outputs)
tr = np.arange(0, dt * len(cr), dt)
mask_crs = protocols.mask(tr, step_duration, discard=remove,
                          discard_start=discard_start)
tr = tr[mask_crs]
cr = cr[mask_crs]
crs = crs[mask_crs, :]

'''
import matplotlib.pyplot as plt
plt.plot(model.times(), model.simulate(parameters_true)[:, 1], '.')
#plt.plot(tr, crs[:, 1], 'x')
plt.plot(model.times(), crs[:, 1], 'x')
plt.show()
sys.exit()
#'''

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
    guess[-2] += 9 # TODO voff
    print('Initial guess:', guess)
else:
    guess = None

# Create boundaries and transformation
class LogRectBounds(pints.RectangularBoundaries):
    """
    Rect boundaries, but samples around 1.
    """
    def set_non_scaling_parameters(self, x):
        """ Set non-scaling parameter indices. """
        self._non_scaling_parameters = x

    def sample(self, n=1):
        """ See :meth:`pints.Boundaries.sample()`. """
        xs = 1.
        xs += np.random.normal(0, 0.1, size=(n, self._n_parameters))
        xs[:, self._non_scaling_parameters] -= 1
        xs[:, -2] += 10 # TODO voff
        return xs

b_scale = 1e5
b_value = 30
def setup_bound_and_transform(model, b_scale, b_value):
    """
    Setup and return boundaries and transformation.
    """
    # Check if we have voltage offset in the parameters
    p = model.fit_parameter_names()
    n_parameters = model.n_parameters()
    assert(len(p) == n_parameters)
    l = np.ones(n_parameters) / b_scale
    u = np.ones(n_parameters) * b_scale
    l[model.non_scaling_parameters()] = -1 * b_value
    u[model.non_scaling_parameters()] = b_value
    l[-2] = 5 # TODO voff
    print(f'Total {model.n_parameters()} parameters.')
    print(f'Parameters {model.non_scaling_parameters()} are non-scaling.')
    # Create boundaries
    boundaries = LogRectBounds(l, u)
    boundaries.set_non_scaling_parameters(model.non_scaling_parameters())
    # Create transformation for bounded parameters
    transformation = pints.RectangularBoundariesTransformation(boundaries)
    return boundaries, transformation

boundaries, transformation = setup_bound_and_transform(model, b_scale, b_value)

# Create score function
if len(pnames) > 1:
    problem = pints.MultiOutputProblem(model, tr, crs)
else:
    problem = pints.SingleOutputProblem(model, tr, cr)
if not ENFORCE_SIMILAR:
    error = pints.MeanSquaredError(problem)
else:
    error = pints.SumOfSquaresError(problem)
    # Fit ideal to 80% CP data too to enforce the ideal is not too far off.
    ideal = models.VCModel(
        models.mmt(mname),
        fit_kinetics=FIT_KINETICS,
        vc_level=models.VC_IDEAL,
        temperature=temperature,
    )
    ideal.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold,
                       mask=mask)
    problem
    p = pints.SingleOutputProblem(ideal, tr, cr)  # TODO check is 80CP
    e = pints.SumOfSquaresError(p)
    # TODO hack a bit...
    e._n_parameters = error._n_parameters
    error = pints.SumOfErrors([error, e])


# Try fitting
path = os.path.join(results, ename)
utils.fit(path, error, boundaries, transformation, 10, 50, guess=guess)

# Show current best results
parameters, info = utils.load(
    os.path.join(path, 'result.txt'), n_parameters=n_parameters)
utils.show_summary(parameters, info)


#!/usr/bin/env python3
#
# Fit a model to a current trace.
#
#!/usr/bin/env python3
import os

import numpy as np
import matplotlib.pyplot as plt
import pints

from methods import data2 as data
from methods import utils, models, protocols
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
discard = 0 # 2000  # ms
v_steps = data._naiv_steps
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
    #alphas=alphas if level != models.VC_IDEAL else None,
    E_leak=True,
    temperature=temperature,
)
model.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold)
mask = protocols.mask(model.times(), step_duration, discard=discard)
model.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold, mask=mask)

ideal = models.VCModel(
    models.mmt(mname),
    fit_kinetics=True,
    vc_level=models.VC_IDEAL,
)
ideal.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold, mask=mask)

# Create parameter vector
n_parameters = model.n_parameters()
parameters_true = np.ones(n_parameters)

# Set voltage clamp setting
data.setup_model_vc(dname, model)

# Show current best results
path = os.path.join(results, ename)
parameters, info = utils.load(
    os.path.join(path, 'result.txt'), n_parameters=n_parameters)
utils.show_summary(parameters, info)

# Generate or load data
fig = plt.figure()
fig.set_figheight(12)
fig.set_figwidth(20)
ax_iv_1 = plt.subplot2grid((2, 6), (0, 3), colspan=3, rowspan=1)
ax_iv_1.set_xticklabels([])
ax_iv_2 = plt.subplot2grid((2, 6), (1, 3), colspan=3, rowspan=1)

if not os.path.isdir(os.path.join(results, f'prediction-raw-{ename}')):
    os.makedirs(os.path.join(results, f'prediction-raw-{ename}'))

# Ideal
ir = ideal.simulate(parameters[0][:ideal.n_parameters()])
ti, ci = protocols.fold(ideal.times(), ir, step_duration, discard=discard)
ii_iv, vi_iv = protocols.naiv_iv(ti, ci, is_data=False)
ax_iv_1.plot(vi_iv, ii_iv, c='k', ls=':', label=f'ideal')
ax_iv_2.plot(vi_iv, ii_iv, c='k', ls=':', label=f'ideal')
ax_0 = plt.subplot2grid(shape=(2, 6), loc=(1, 2))
for j, v in enumerate(v_steps):
    ax_0.plot(ti, ci[j],
              ls=':', label='__' if j else 'ideal', alpha=0.5)

axes = []
ymin = np.inf
ymax = -np.inf
for ii, i in enumerate(range(0, 9, 2)):
    ax = plt.subplot2grid(shape=(2, 6), loc=(int(ii/3), ii%3))
    axes.append(ax)
    fig_i, ax_i = plt.subplots(1, 1)

    pname = f'NaIV_35C_{i*10}CP'
    tr, vr_d, cr_d = data.load_named(dname,
                                     pname,
                                     model,
                                     parameters_true,
                                     shift=True)
    model.set_artefact_parameters({
        'voltage_clamp.alpha_R': i / 10.,
        'voltage_clamp.alpha_P': i / 10.,
    })
    mr = model.simulate(parameters[0])
    tm, cm = protocols.fold(model.times(), mr, step_duration, discard=discard)

    for j, v in enumerate(v_steps):
        # Individual plot
        ax_i.plot(tr, cr_d[v], label='__' if j else 'data')
        ax_i.plot(tm, cm[j],
                  c=ax_i.get_lines()[-1].get_color(),
                  ls='--', label='__' if j else 'fitted')
        ax_i.plot(ti, ci[j],
                  c=ax_i.get_lines()[-1].get_color(),
                  ls=':', label='__' if j else 'ideal', alpha=0.5)

        # Big plot
        ax.plot(tr, cr_d[v], label='__' if j else 'data', alpha=0.75)
        ax.plot(tm, cm[j],
                c=ax.get_lines()[-1].get_color(),
                ls='--', label='__' if j else 'fitted')
        if int(i // 3) < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (ms)')
        if i % 3 != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Current (pA)')
        ymin = min(ymin, np.min(cr_d[v]))
        ymin = min(ymin, np.min(cm[j]))
        ymax = max(ymax, np.max(cr_d[v]))
        ymax = max(ymax, np.max(cm[j]))

    # IV plot
    ir_iv, vr_iv = protocols.naiv_iv(tr, cr_d, is_data=True)
    im_iv, vm_iv = protocols.naiv_iv(tm, cm, is_data=False)
    ax_iv_1.plot(vr_iv, ir_iv, label=f'CP{i*10}')
    ax_iv_2.plot(vm_iv, im_iv,
               c=ax_iv_1.get_lines()[-1].get_color(),
               ls='--', label=f'alpha={i/10.}')

    # Individual plot tidy
    title = ' '.join([mname,
                      f'vc_level {level}',
                      ' '.join(pnames),
                      dname,])
    if (i/10., i/10.) in alphas:
        print(i, title)
        ax_i.set_title(f'Fitting results for CP{i*10} with {title}')
    else:
        ax_i.set_title(f'Prediction results for CP{i*10} with {title}')
    ax_i.legend()
    ax_i.set_xlim([9.5, 14.5])
    fig_i.tight_layout()
    path = os.path.join(results, f'prediction-raw-{ename}', f'cp{i*10}.png')
    print(f'Writing figure to {path}')
    fig_i.savefig(path)
    plt.close(fig_i)

for ax in axes:
    ax.set_ylim([ymin-100, ymax+100])
    ax.set_xlim([9.5, 14])
ax_iv_1.legend()
ax_iv_2.legend()
ax_iv_2.set_xlabel('Voltage (mV)')
ax_iv_1.set_ylabel('Current (pA)')
ax_iv_2.set_ylabel('Current (pA)')
fig.tight_layout()
path = os.path.join(results, f'prediction-all-{ename}.png')
print(f'Writing figure to {path}')
fig.savefig(path)
plt.close(fig)

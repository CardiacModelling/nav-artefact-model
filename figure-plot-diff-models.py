#!/usr/bin/env python3
#
# Fit a model to a current trace.
#
#!/usr/bin/env python3
import os

import numpy as np

from methods import data2 as data
from methods import utils, models, protocols
from methods import results, run, t_hold, v_hold

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

colour_list = sns.color_palette('coolwarm', n_colors=5).as_hex()
colour_vlist = sns.color_palette('viridis', n_colors=15).as_hex()

# Get model name, vc level, protocol name, data name, and experiment name
mname, level, pnames, dname, ename = utils.cmd('Perform a fit')
FIT_KINETICS = not True
FIT_ARTEFACT = not True
USE_IV_FIT = not True

figures = 'figures'

mname2 = 'iyer'

#TODO
pnames = [pnames[0]]

LJP = 9  # mV; convention: Vm = Vcmd - LJP

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
    discard_start = 0
    remove = 0 + discard_start
    step_duration = 40 - remove  # ms
    discard = remove + 2000  # ms
    v_steps = data._naiv(dname)
    protocol = protocols.load('protocols/ina-steps.txt')
elif dname in data.batch2:
    discard_start = 0
    remove = 0 + discard_start
    step_duration = 40 - remove  # ms
    discard = remove + 0  # ms
    v_steps = data._naiv(dname)
    protocol = protocols.load('protocols/ina-steps-2-no-holding.txt')
elif dname in data.batch3 + data.batch4:
    discard_start = 0
    remove = 0 + discard_start
    step_duration = 39 - remove  # ms
    discard = remove + 2000  # ms
    v_steps = data._naiv(dname)
    protocol = protocols.load('protocols/ina-steps-3.txt')

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
    fit_artefacts=FIT_ARTEFACT,
    vc_level=level,
    #alphas=alphas if level != models.VC_IDEAL else None,
    E_leak=True,
    temperature=temperature,
)
model._artefact_vars += ['voltage_clamp.tau_out']
model.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold)
mask = protocols.mask(model.times(), step_duration, discard=discard)
model.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold, mask=mask)

model2 = models.VCModel(
    models.mmt(mname2),
    fit_kinetics=FIT_KINETICS,
    fit_artefacts=FIT_ARTEFACT,
    vc_level=level,
    #alphas=alphas if level != models.VC_IDEAL else None,
    E_leak=True,
    temperature=temperature,
)
model2._artefact_vars += ['voltage_clamp.tau_out']
model2.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold, mask=mask)

ideal = models.VCModel(
    models.mmt(mname),
    fit_kinetics=FIT_KINETICS,
    vc_level=models.VC_IDEAL,
    temperature=temperature,
)
protocol_ljp = np.loadtxt('protocols/ina-steps-3.txt')
protocol_ljp[:, 0] -= LJP
protocol_ljp = protocols.from_steps(protocol_ljp.flatten())
#ideal.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold, mask=mask)
ideal.set_protocol(protocol_ljp, dt=dt, v_hold=v_hold, t_hold=t_hold, mask=mask)

ideal2 = models.VCModel(
    models.mmt(mname2),
    fit_kinetics=FIT_KINETICS,
    vc_level=models.VC_IDEAL,
    temperature=temperature,
)
ideal2.set_protocol(protocol_ljp, dt=dt, v_hold=v_hold, t_hold=t_hold, mask=mask)

# Create parameter vector
n_parameters = model.n_parameters()
parameters_true = np.ones(n_parameters)

# Set voltage clamp setting
data.setup_model_vc(dname, model, ljp=LJP)

# Show current best results
'''
name = f'results-test'
name += f'-{mname}'
name += f'-vc{level}'
name += f'-cell6'
name += f'-{"-".join(pnames)}'
path = os.path.join(results, name)
'''
path = os.path.join(results, ename)
print(f'Loading results from {path}')
parameters, info = utils.load(
    os.path.join(path, 'result.txt'), n_parameters=n_parameters)
utils.show_summary(parameters, info)
if USE_IV_FIT:
    conductance, _ = utils.load(os.path.join(
        results,
        f'results-test-{mname}-vc90-{dname}-NaIV_35C_0CP-NaIV_35C_80CP',
        'result.txt'), n_parameters=1)
    print('Without IV:', parameters[0])
    parameters[0][0] = conductance[0]
    print('With IV:', parameters[0])
print('Using parameters:', parameters[0])

ir = ideal.simulate(parameters[0][:ideal.n_parameters()])
ti, ci = protocols.fold(ideal.times(), ir, step_duration, discard=discard)
irv = ideal.voltage(parameters[0][:ideal.n_parameters()])
tiv, vi = protocols.fold(ideal.times(), irv, step_duration, discard=discard)

ename2 = ename.replace(mname, mname2)
path2 = os.path.join(results, ename2)
print(f'Loading results from {path2}')
parameters2, info2 = utils.load(
    os.path.join(path2, 'result.txt'), n_parameters=n_parameters)
utils.show_summary(parameters2, info2)
if len(parameters2) == 0:
    ir2 = ideal2.simulate(parameters[0][:ideal2.n_parameters()])
    ti2, ci2 = protocols.fold(ideal2.times(), ir2, step_duration, discard=discard)
    ii_iv, vi_iv = protocols.naiv_iv(ti, ci, dname, is_data=False)
    ii_iv2, vi_iv2 = protocols.naiv_iv(ti2, ci2, dname, is_data=False)
    parameters2 = [[parameters[0][0] * (np.min(ii_iv) / np.min(ii_iv2))]]
    print('Using ratio results', parameters2[0])
else:
    print('Using parameters', parameters2[0])

###############################################################################
# Generate or load data
fig = plt.figure()
fig.set_figheight(6.5)
fig.set_figwidth(10)
alpha_all = [0, 20, 40, 60, 80]
n_grids = (9, 15)

# Top row
axes_top = []
for i, alpha in enumerate(alpha_all):
    ax = plt.subplot2grid(n_grids, (0, 3*i), colspan=3, rowspan=3, fig=fig)
    axes_top.append(ax)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if i != 0:
        ax.axes.get_yaxis().set_visible(False); ax.spines['left'].set_visible(False)

    pname = f'NaIV_{int(temperature - 273.15)}C_{alpha}CP'
    try:
        tr, vr_d, cr_d = data.load_named(dname,
                                         pname,
                                         model,
                                         parameters_true,
                                         shift=True)
        has_data = True
    #except (FileNotFoundError, OSError) as e:
    #except as e:
    except OSError as e:
        print(e)
        has_data = False

    if has_data:
        for v, c in zip(vr_d, colour_vlist):
            ax.plot(tr, cr_d[v], c=c, alpha=0.75)

    ax.set_title(r'$\alpha$'f'={alpha}%', fontsize=10)
    if dname in data.batch3 + data.batch4:
        ax.set_xlim([9, 11])
    else:
        ax.set_xlim([9.5, 14])
    ax.set_xticks([9, 11])
    ax.set_xticklabels([])
    ax.set_ylim([-34000, 1000])
    ax.set_yticks([-34000, 1000])
    ax.set_yticklabels([])

fig.tight_layout(rect=(0.05, 0.05, 1, 1))  # Adjust layout here with all axes empty

for ax in axes_top:
    ax.set_xlabel('Time (2 ms)')
axes_top[0].set_ylabel('Current (35 nA)')


# Bottom row
v = -40; v_i = np.where(np.sort(list(vr_d.keys())) == v)[0][0]
axes_bottom = []

# Bottom row: data
ax_v = plt.subplot2grid(n_grids, (5, 0), colspan=5, rowspan=2, fig=fig)
ax_i = plt.subplot2grid(n_grids, (7, 0), colspan=5, rowspan=2, fig=fig)
ax_v.set_title('Data', fontsize=10, loc='left')
axes_bottom.append(ax_v); axes_bottom.append(ax_i)
ax_v.step(tiv, vi[v_i]+LJP, ls=':', c='#7f7f7f') # where='post' # Voltage
for i, alpha in enumerate(alpha_all):
    pname = f'NaIV_{int(temperature - 273.15)}C_{alpha}CP'
    try:
        tr, vr_d, cr_d = data.load_named(dname,
                                         pname,
                                         model,
                                         parameters_true,
                                         shift=True)
        has_data = True
    #except (FileNotFoundError, OSError) as e:
    #except as e:
    except OSError as e:
        print(e)
        has_data = False

    if has_data:
        ax_i.plot(tr, cr_d[v], c=colour_list[i])

# Bottom row: model
ax_v = plt.subplot2grid(n_grids, (5, 5), colspan=5, rowspan=2, fig=fig)
ax_i = plt.subplot2grid(n_grids, (7, 5), colspan=5, rowspan=2, fig=fig)
ax_v.set_title('Model: Gray et al. (2020)', fontsize=10, loc='left')
axes_bottom.append(ax_v); axes_bottom.append(ax_i)
model.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold, mask=mask)
for i, alpha in enumerate(alpha_all):
    c = colour_list[i]
    ax_v.step(tiv, vi[v_i]+LJP, ls=':', c='#7f7f7f') # where='post' # Voltage

    model.set_artefact_parameters({
        'voltage_clamp.alpha_R': alpha / 100.,
        'voltage_clamp.alpha_P': alpha / 100.,
        'voltage_clamp.tau_out': 7.5e-3,
    })
    mr = model.simulate(parameters[0])
    tm, cm = protocols.fold(model.times(), mr, step_duration, discard=discard)
    vr = model.voltage(parameters[0])
    tvm, vm = protocols.fold(model.times(), vr, step_duration, discard=discard)

    ax_i.plot(tm, cm[v_i], ls='--', c=c)
    ax_v.plot(tvm, vm[v_i], ls='--', c=c) # Voltage

# Bottom row: model2
ax_v = plt.subplot2grid(n_grids, (5, 10), colspan=5, rowspan=2, fig=fig)
ax_i = plt.subplot2grid(n_grids, (7, 10), colspan=5, rowspan=2, fig=fig)
ax_v.set_title('Model: Iyer et al. (2004)', fontsize=10, loc='left')
axes_bottom.append(ax_v); axes_bottom.append(ax_i)
model2.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold, mask=mask)
for i, alpha in enumerate(alpha_all):
    c = colour_list[i]
    ax_v.step(tiv, vi[v_i]+LJP, ls=':', c='#7f7f7f') # where='post' # Voltage

    model2.set_artefact_parameters({
        'voltage_clamp.alpha_R': alpha / 100.,
        'voltage_clamp.alpha_P': alpha / 100.,
        'voltage_clamp.V_offset_eff': -LJP,
        'voltage_clamp.tau_out': 7.5e-3,
    })
    mr = model2.simulate(parameters2[0])
    tm, cm = protocols.fold(model2.times(), mr, step_duration, discard=discard)
    vr = model2.voltage(parameters2[0])
    tvm, vm = protocols.fold(model2.times(), vr, step_duration, discard=discard)

    ax_i.plot(tm, cm[v_i], ls='--', c=c)
    ax_v.plot(tvm, vm[v_i], ls='--', c=c) # Voltage


for i, ax in enumerate(axes_bottom):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if dname in data.batch3 + data.batch4:
        ax.set_xlim([9, 12.5])
    else:
        ax.set_xlim([9.5, 14])
    ax.set_xticks([9, 12.5])
    ax.set_xticklabels([])
    if i % 2:
        ax.set_ylim([-24000, 1000])
        ax.set_yticks([-24000, 1000])
        ax.set_yticklabels([])
        ax.set_xlabel('Time (3.5 ms)')
    else:
        ax.set_ylim([-110, 40])
        ax.set_yticks([-110, 40])
        ax.set_yticklabels([])
        ax.axes.get_xaxis().set_visible(False); ax.spines['bottom'].set_visible(False)
    if i == 0:
        ax.set_ylabel('Voltage (150 mV)')
    elif i == 1:
        ax.set_ylabel('Current (25 nA)')
    else:
        ax.axes.get_yaxis().set_visible(False); ax.spines['left'].set_visible(False)

axes_bottom[0].text(11.8, -35, r'$-40$ mV')
axes_bottom[0].text(9.2, -105, r'$-100$ mV')

axes_top[0].annotate(r'$-40$ mV', xy=(9.65, -20000), xytext=(9.1, -31500),
            arrowprops=dict(arrowstyle="->", color='red'))
axes_top[1].annotate(r'$-40$ mV', xy=(9.65, -23000), xytext=(9.75, -32000),
            arrowprops=dict(arrowstyle="->", color='red'))
axes_top[2].annotate(r'$-40$ mV', xy=(10.5, -22000), xytext=(9.75, -31000),
            arrowprops=dict(arrowstyle="->", color='red'))
axes_top[3].annotate(r'$-30$ mV', xy=(9.75, -29000), xytext=(10.25, -24000),
            arrowprops=dict(arrowstyle="->", color='red'))
axes_top[4].annotate(r'$-30$ mV', xy=(9.8, -4000), xytext=(10., -13000),
            arrowprops=dict(arrowstyle="->", color='red'))

axes_top[0].text(-0.3, 1.1, 'A', transform=axes_top[0].transAxes,
                ha='center', va='center', fontsize=12, fontweight='bold')
axes_bottom[0].text(-0.3*3/5, 1.65, 'B', transform=axes_bottom[0].transAxes,
                ha='center', va='center', fontsize=12, fontweight='bold')

# Inset protocol
ax_protocol = axes_top[0].inset_axes([0.5, 0.05, 0.5, 0.7])
ax_protocol.set_xticks([]); ax_protocol.set_yticks([])
ax_protocol.set_xticklabels([]); ax_protocol.set_yticklabels([])
ax_protocol.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
ax_protocol.axvspan(9, 12.5, color='gray', alpha=0.25)
for i in range(len(vi)):
    c = colour_vlist[i]
    ax_protocol.step(tiv, vi[i]+LJP, c=c) # where='post' # Voltage

# Colorbar
#fig.subplots_adjust(top=0.945)
cbar_ax = fig.add_axes([0.085, 0.525, 0.885, 0.0325])
cmap = ListedColormap(colour_list)
cbar = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                        orientation='horizontal')
cbar.ax.get_xaxis().set_ticks([])
for j, x in enumerate([0, 20, 40, 60, 80]):
    cbar.ax.text((2 * j + 1) / (2 * 5), .5,
                #f'{int(alpha_r*100)}%, {int(alpha_p*100)}%',
                r'$\alpha$'f'={int(x)}%',
                ha='center', va='center', fontsize=10)

# Save figure
path = os.path.join(figures, f'figure-{mname2}-{ename}')
print(f'Writing figure to {path}')
fig.savefig(path, dpi=300)
fig.savefig(path + '.pdf', format='pdf')
plt.close(fig)

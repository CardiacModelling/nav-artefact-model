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

# Get model name, vc level, protocol name, data name, and experiment name
mname, level, pnames, dname, ename = utils.cmd('Perform a fit')
FIT_KINETICS = not True
FIT_ARTEFACT = True
USE_IV_FIT = not True

figures = 'figures'

ARROW = True

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
model.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold)
mask = protocols.mask(model.times(), step_duration, discard=discard)
model.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold, mask=mask)

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

# Create long VC model
protocol_long = [[-100., 9.1], [-80., 20.], [-100., 10.]]
s = 7.5  # mV
vs_long = np.arange(-80, 60 + s, s)
t_hold_long = 200 # ms
for i in vs_long[1:]:
    protocol_long.append([-100., t_hold_long])
    protocol_long.append([-100, 9.])
    protocol_long.append([i, 20.])
    protocol_long.append([-100, 10.])
protocol_ljp_long = np.array(protocol_long).copy()
protocol_ljp_long[:, 0] -= LJP
protocol_long = protocols.from_steps(np.array(protocol_long).flatten())
protocol_ljp_long = protocols.from_steps(protocol_ljp_long.flatten())
model.set_protocol(protocol_long, dt=dt, v_hold=v_hold, t_hold=t_hold_long)
mask_long = protocols.mask(model.times(), step_duration, discard=t_hold_long)
#model.set_protocol(protocol_long, dt=dt, v_hold=v_hold, t_hold=t_hold_long, mask=mask_long)
model.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold, mask=mask)

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

###############################################################################
# Generate or load data
fig = plt.figure()
fig.set_figheight(8)
fig.set_figwidth(12)
axes_v1 = []
axes_i1 = []
axes_v2 = []
axes_i2 = []

# IV curves
ax_iv_1 = plt.subplot2grid((14, 10), (0, 0), colspan=4, rowspan=7, fig=fig)
ax_iv_1.set_xticklabels([])
ax_iv_2 = plt.subplot2grid((14, 10), (7, 0), colspan=4, rowspan=7, fig=fig)
fig.tight_layout(rect=(0.05, 0.05, 1, 1))

# Ideal
print(ideal.fit_parameter_names(), ideal.n_parameters())
print(parameters[0])
ideal.set_protocol(protocol_ljp_long, dt=dt, v_hold=v_hold, t_hold=t_hold_long, mask=mask_long)
ir = ideal.simulate(parameters[0][:ideal.n_parameters()])
ti, ci = protocols.fold(ideal.times(), ir, step_duration, discard=t_hold_long)
ii_iv, vi_iv = protocols.naiv_iv(ti, ci, dname, is_data=False, vs=vs_long)
ideal.set_protocol(protocol_ljp, dt=dt, v_hold=v_hold, t_hold=t_hold, mask=mask)
ir = ideal.simulate(parameters[0][:ideal.n_parameters()])
ti, ci = protocols.fold(ideal.times(), ir, step_duration, discard=discard)
irv = ideal.voltage(parameters[0][:ideal.n_parameters()])
tiv, vi = protocols.fold(ideal.times(), irv, step_duration, discard=discard)
#ax_iv_1.plot(vi_iv, ii_iv, c='k', ls=':', label=f'ideal')
ax_iv_2.plot(vi_iv, ii_iv, c='k', ls=':', label=f'ideal')
ax_iv_1.axvline(x=-30, c='#7f7f7f', ls=':')
ax_iv_1.axvline(x=-10, c='#7f7f7f', ls='--')
ax_iv_2.axvline(x=-30, c='#7f7f7f', ls=':')
ax_iv_2.axvline(x=-10, c='#7f7f7f', ls='--')

# Create plots for -30 and -10 mV (index 5 and 7)
for i, alpha in enumerate([0, 20, 40, 60, 80]):
    c = colour_list[i]

    axv1 = plt.subplot2grid(shape=(14, 10), loc=(1, i+5), rowspan=3, fig=fig)
    axi1 = plt.subplot2grid(shape=(14, 10), loc=(4, i+5), rowspan=3, fig=fig)
    axv2 = plt.subplot2grid(shape=(14, 10), loc=(8, i+5), rowspan=3, fig=fig)
    axi2 = plt.subplot2grid(shape=(14, 10), loc=(11, i+5), rowspan=3, fig=fig)

    # Command voltage
    axv1.step(tiv, vi[5]+LJP, ls=':', c='#7f7f7f') # where='post' # Voltage
    axv2.step(tiv, vi[7]+LJP, ls=':', c='#7f7f7f') # where='post' # Voltage

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
    model.set_artefact_parameters({
        'voltage_clamp.alpha_R': alpha / 100.,
        'voltage_clamp.alpha_P': alpha / 100.,
    })
    mr = model.simulate(parameters[0])
    tm, cm = protocols.fold(model.times(), mr, step_duration, discard=discard)
    vr = model.voltage(parameters[0])
    tvm, vm = protocols.fold(model.times(), vr, step_duration, discard=discard)

    # Model long
    #model_long.set_artefact_parameters({
    #    'voltage_clamp.alpha_R': alpha / 100.,
    #    'voltage_clamp.alpha_P': alpha / 100.,
    #})
    model.set_protocol(protocol_long, dt=dt, v_hold=v_hold, t_hold=t_hold_long, mask=mask_long)
    mr_long = model.simulate(parameters[0])
    tm_long, cm_long = protocols.fold(model.times(), mr_long, step_duration, discard=t_hold_long)
    model.set_protocol(protocol, dt=dt, v_hold=v_hold, t_hold=t_hold, mask=mask)

    if has_data:
        axi1.plot(tr, cr_d[-30], label='Data', c=c)
    axi1.plot(tm, cm[5],
                c=axi1.get_lines()[-1].get_color(),
                ls='--', label='Model prediction')
    axv1.plot(tvm, vm[5], ls='--', c=c) # Voltage

    if has_data:
        axi2.plot(tr, cr_d[-10], label='Data', c=c)
    axi2.plot(tm, cm[7],
                c=axi2.get_lines()[-1].get_color(),
                ls='--', label='Model prediction')
    axv2.plot(tvm, vm[7], ls='--', c=c) # Voltage

    # Ideal
    axi1.plot(ti, ci[5], ls=':', label='Physiological (theoretical)', c='#7f7f7f')
    axi2.plot(ti, ci[7], ls=':', label='Physiological (theoretical)', c='#7f7f7f')

    # IV plot
    if has_data:
        ir_iv, vr_iv = protocols.naiv_iv(tr, cr_d, dname, is_data=True)
        ax_iv_1.plot(vr_iv, ir_iv, marker='x', ls='', label=f'CP{alpha}', c=c)
        ax_iv_1.plot(vr_iv, ir_iv, ls='-', c=c, alpha=0.25)
    im_iv, vm_iv = protocols.naiv_iv(tm, cm, dname, is_data=False)
    ax_iv_2.plot(vm_iv, im_iv, c=c, marker='o', ls='', label=f'alpha={alpha}%')
    im_iv_long, vm_iv_long = protocols.naiv_iv(tm_long, cm_long, dname, is_data=False, vs=vs_long)
    #ax_iv_2.plot(vm_iv_long, im_iv_long, c=c, ls='--', alpha=0.25)
    ax_iv_2.plot(vm_iv, im_iv, c=c, ls='--', alpha=0.25)
    
    
    # Tidy up
    axv1.set_title(r'$\alpha$'f'={alpha}%')
    axv1.spines['top'].set_visible(False)
    axv1.spines['right'].set_visible(False)
    axi1.spines['top'].set_visible(False)
    axi1.spines['right'].set_visible(False)
    axv2.set_title(r'$\alpha$'f'={alpha}%')
    axv2.spines['top'].set_visible(False)
    axv2.spines['right'].set_visible(False)
    axi2.spines['top'].set_visible(False)
    axi2.spines['right'].set_visible(False)

    if i != 0:
        #axv1.set_yticklabels([])
        axv1.axes.get_yaxis().set_visible(False); axv1.spines['left'].set_visible(False)
        axi1.axes.get_yaxis().set_visible(False); axi1.spines['left'].set_visible(False)
        axv2.axes.get_yaxis().set_visible(False); axv2.spines['left'].set_visible(False)
        axi2.axes.get_yaxis().set_visible(False); axi2.spines['left'].set_visible(False)
    else:
        axv1.set_ylabel('Voltage (mV)', fontsize=13)
        axi1.set_ylabel('Current (pA)', fontsize=13)
        axv2.set_ylabel('Voltage (mV)', fontsize=13)
        axi2.set_ylabel('Current (pA)', fontsize=13)

    axes_v1.append(axv1)
    axes_i1.append(axi1)
    axes_v2.append(axv2)
    axes_i2.append(axi2)


# Big plot tidy
for ax in axes_v1 + axes_i1 + axes_v2 + axes_i2:
    #ax.set_ylim([ymin-100, ymax+100])
    if dname in data.batch3 + data.batch4:
        ax.set_xlim([8.9, 10])
    else:
        ax.set_xlim([9.5, 14])
    ax.set_xticks([8.9, 10])
    ax.set_xticklabels([])
for ax in axes_v1 + axes_v2:
    #ax.set_xticklabels([])
    ax.set_ylim([-125, 35])
for ax in axes_i1 + axes_i2:
    ax.set_ylim([-38000, 1000])
    ax.set_xlabel('Time (1 ms)')

#ax_iv_1.legend()
#ax_iv_2.legend()
ax_iv_1.spines['top'].set_visible(False)
ax_iv_1.spines['right'].set_visible(False)
ax_iv_2.spines['top'].set_visible(False)
ax_iv_2.spines['right'].set_visible(False)
ax_iv_1.set_ylim([-40000, 1000])
ax_iv_2.set_ylim([-40000, 1000])
ax_iv_2.set_xlabel('Intended test-pulse voltage (mV)', fontsize=13)
ax_iv_1.set_ylabel('Data current (pA)', fontsize=13)
ax_iv_2.set_ylabel('Model current (pA)', fontsize=13)
fig.align_ylabels([ax_iv_1, ax_iv_2])
fig.align_ylabels(axes_v1 + axes_i1 + axes_v2 + axes_i2)

ax_iv_1.text(-0.15, 0.95, 'i', transform=ax_iv_1.transAxes,
                ha='center', va='center', fontsize=12, fontweight='bold')
ax_iv_2.text(-0.15, 0.95, 'ii', transform=ax_iv_2.transAxes,
                ha='center', va='center', fontsize=12, fontweight='bold')
axes_v1[0].text(-0.5, 1.3, 'iii', transform=axes_v1[0].transAxes,
                ha='center', va='center', fontsize=12, fontweight='bold')
axes_v2[0].text(-0.5, 1.3, 'iv', transform=axes_v2[0].transAxes,
                ha='center', va='center', fontsize=12, fontweight='bold')

axes_i1[-1].legend(bbox_to_anchor=(1.1, 1.2), loc="lower right", ncol=3,
                   bbox_transform=axes_v1[-1].transAxes)

if ARROW:
    #ax_iv_1.annotate(
    #    '', xy=(25, -35000), xytext=(-15, -2500),
    #    arrowprops=dict(arrowstyle='->', lw=2, color='C2', alpha=0.75)
    #)
    #ax_iv_2.annotate(
    #    '', xy=(25, -35000), xytext=(-15, -2500),
    #    arrowprops=dict(arrowstyle='->', lw=2, color='C2', alpha=0.75)
    #)
    ax_iv_2.text(22.5, -31500, 'Physiological\n(theoretical with\nno artifacts)', fontsize=12, ha='left')

# Colorbar
fig.subplots_adjust(top=0.945)
cbar_ax = fig.add_axes([0.09, 0.96, 0.9, 0.0325])
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
path = os.path.join(figures, f'figure-{ename}')
print(f'Writing figure to {path}')
fig.savefig(path, dpi=300)
fig.savefig(path + '.pdf', format='pdf')
plt.close(fig)

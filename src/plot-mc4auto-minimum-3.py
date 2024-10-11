#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import util

import model as m; m.vhold = -40

"""
Plot model cell experiment data with amplifier settings.
"""

saveas = 'mc4auto-minimum'

#sim_list = ['staircase', 'sinewave', 'ap-beattie', 'ap-lei']
sim_list = ['nav']
data_idx = {'staircase': 2, 'sinewave': 1, 'ap-lei': 3, 'nav': 0, 'ramps':4}
protocol_list = {
    'nav': [lambda x: (x, np.loadtxt('nav-step.txt'))],
#    'ramps': ['ramps/ramps_%s.csv' % i for i in range(10)],
#    'staircase': ['staircase-ramp.csv'],
#    'sinewave': ['sinewave-ramp.csv'],
#    'ap-beattie': ['ap-beattie.csv'],
#    'ap-lei': ['ap-lei.csv'],
}
legend_ncol = {
    'nav': (2, 1),
#    'ramps': (2, 1),
#    'staircase': (2, 1),
#    'sinewave': (1, 1),
#    'ap-beattie': (4, 2),
#    'ap-lei': (4, 2),
}

wins = {
    'nav': np.array([45, 80]),
#    'ramps': np.array([45, 150]),
}

try:
    which_sim = sys.argv[1]
except:
    print('Usage: python %s [str:which_sim]' % os.path.basename(__file__))
    sys.exit()

if which_sim not in sim_list:
    raise ValueError('Input data %s is not available in the predict list' \
            % which_sim)

savedir = '.'
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# Check the lab record (docx) for details
USE95 = not False
if USE95:
    n_experiments = {
        0: [0, 1, 2, 3, 4, 5],
        1: [0, 6, 7, 8, 9, 10],
        2: [10, 11, 12, 13, 14, 15],
    }
else:
    n_experiments = {
        0: [0, 1, 2, 3, 4],
        1: [0, 6, 7, 8, 9],
        2: [0, 11, 12, 13, 14],
    }

f = '../data/mc/20230510-Cfast_auto.dat'
#f = '../data/mc/20230510.dat'
n_sweeps = 10 if which_sim == 'ramps' else 1

# Model setup
parameters = [
    'mc.gk',
    'mc.ck',
    'mc.gm',
    'voltageclamp.C_prs',
    'mc.Cm',
    'voltageclamp.R_series',
    'voltageclamp.V_offset_eff',
    'voltageclamp.C_prs_est',
    'voltageclamp.Cm_est',
    'voltageclamp.R_series_est',
    'voltageclamp.alpha_R',
    'voltageclamp.alpha_P',
    'voltageclamp.tau_out',
]

# What we have in the circuit
p = [
    1./0.01,  # pA/mV = 1/GOhm; g_kinetics
    100.,  # pF; C_kinetics
    1./0.1,  # pA/mV = 1/GOhm; g_membrane
    7.2,#0.0001,#4.7,  # pF; Cprs
    44.,  # pF; Cm
    30e-3,  # GOhm; Rs
    0,  # mV; Voffset+
    7.2,#0.0001,#7.2,  # pF; Cprs*
    44.,  # pF; Cm*
    30.04e-3,  # GOhm; Rs*
    None,  # alpha_R
    None,  # alpha_P
    110e-3, # tau_out: 110e-3ms
    # For alpha=95%, probably need tau_out=155e-3ms and 30% reduction in Cm*?
]
i_alpha_r, i_alpha_p = -3, -2

extra_log = ['voltageclamp.Vc', 'membrane.V']

models = []
for i_sweep in range(n_sweeps):
    #model = m.Model('full2-voltage-clamp-mc4.mmt',
    models.append(
            m.Model('minimum-voltage-clamp-mc4.mmt',
                    protocol_def=protocol_list[which_sim][i_sweep],
                    temperature=273.15 + 23.0,  # K (doesn't matter here)
                    transform=None,  # Not needed
                    readout='voltageclamp.Iout',
                    useFilterCap=False)
        )
    models[i_sweep].set_parameters(parameters)

# Figure setup
fig, axess = plt.subplots(2, 3, figsize=(12, 5), sharex='all', sharey='row')

n_groups = [1, 2, 0]

# Data setup
for i_col, n_group in enumerate(n_groups):
    axes = axess[:, i_col]

    colour_list = sns.color_palette('coolwarm',
                                    n_colors=len(n_experiments[n_group])).as_hex()

    # Iterate through experiments and load data/simulate/plot
    alphas = []
    for i_experiment, n_experiment in enumerate(n_experiments[n_group]):

        # Get data
        data_ccs, data_vcs, datas = [], [], []
        for i_sweep in range(n_sweeps):
            idx = [n_experiment, data_idx[which_sim], i_sweep]
            whole_data, times = util.load(f, idx, vccc=True)
            data_ccs.append( whole_data[3] * 1e3 )  # V -> mV
            data_vcs.append( whole_data[1] * 1e3 )  # V -> mV
            datas.append( (whole_data[0] + whole_data[2]) * 1e12 )  # A -> pA
        times = times * 1e3  # s -> ms
        alpha_r, alpha_p = util.mc4_experiment_alphas[n_experiment]

        p[i_alpha_r] = alpha_r
        p[i_alpha_p] = alpha_p
        alphas.append((alpha_r, alpha_p))

        # Simulate
        Iouts, Vcs, Vms = [], [], []
        for i_sweep in range(n_sweeps):
            simulation = models[i_sweep].simulate(p, times, extra_log=extra_log)
            Iouts.append( simulation['voltageclamp.Iout'] )
            Vcs.append( simulation['voltageclamp.Vc'] )
            Vms.append( simulation['membrane.V'] )

        # Plot
        c = colour_list[i_experiment]
        ax = axes[0]
        #ax.set_ylabel(r'$\alpha_R=$'f'{alpha_r*100}%\n'r'$\alpha_P=$'f'{alpha_p*100}%')
        times_x = times - wins[which_sim][0]
        for i, (data_vc, data_cc, Vc, Vm) in enumerate(zip(data_vcs, data_ccs, Vcs, Vms)):
            ax.plot(times_x, data_vc, c='#bdbdbd', label='_' if i or i_experiment else r'$V_{cmd}$')
            #ax.plot(times_x, Vc, ls='--', c='#7f7f7f', label='_' if i or i_experiment else r'Input $V_{cmd}$')
            ax.plot(times_x, data_cc, c=c, alpha=0.75, label='_' if i or i_experiment else r'Obs. $V_{m}$')
            ax.plot(times_x, Vm, ls='--', c=c, label='_' if i or i_experiment else r'Sim. $V_{m}$')

        ax = axes[1]
        for i, (data, Iout) in enumerate(zip(datas, Iouts)):
            ax.plot(times_x, data, c=c, label='_' if i or i_experiment else r'Obs. $I_{out}$')
            ax.plot(times_x, Iout, ls='--', c=c, label='_' if i or i_experiment else r'Sim. $I_{out}$')

    if i_col == 0:
        axes[0].set_ylabel('Voltage (mV)', fontsize=13)
        axes[1].set_ylabel('Current (pA)', fontsize=13)
    if i_col == 2:
        axes[0].legend(loc='upper left', #ncol=legend_ncol[which_sim][0],
                bbox_to_anchor=(1.02, 1.), fontsize=10,
                bbox_transform=axes[0].transAxes)
        axes[1].legend(loc='upper left', #ncol=legend_ncol[which_sim][1],
                bbox_to_anchor=(1.02, 1.), fontsize=10,
                bbox_transform=axes[1].transAxes)
    axes[-1].set_xlabel('Time (ms)', fontsize=13)

    fig.align_labels()
    #plt.subplots_adjust(hspace=0)

    if n_group == 0:
        axes[0].set_title(r'Varying $\alpha_R = \alpha_P$')
    elif n_group == 1:
        axes[0].set_title(r'Varying $\alpha_R$ only, $\alpha_P = 0$%')
    elif n_group == 2:
        axes[0].set_title(r'Varying $\alpha_P$ only, $\alpha_R = 95$%')

axes[0].set_xlim(wins[which_sim] - wins[which_sim][0])
plt.tight_layout()

# Colorbar
fig.subplots_adjust(top=0.9)
cbar_ax = fig.add_axes([0.085, 0.95, 0.8, 0.0325])
cmap = ListedColormap(colour_list)
cbar = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                        orientation='horizontal')
cbar.ax.get_xaxis().set_ticks([])
for j, x in enumerate(alphas):
    alpha_r, alpha_p = x
    cbar.ax.text((2 * j + 1) / (2 * len(alphas)), .5,
                #f'{int(alpha_r*100)}%, {int(alpha_p*100)}%',
                f'{int(alpha_r*100)}%',
                ha='center', va='center', fontsize=10)
#cbar.set_label(r'Series resistance compensation ($\alpha_R$) and supercharging ($\alpha_P$)')

plt.savefig('%s/simulate-%s-3-%s-group%s.pdf' % (savedir, saveas, which_sim, n_group), format='pdf')
plt.savefig('%s/simulate-%s-3-%s-group%s' % (savedir, saveas, which_sim, n_group), dpi=300)
#plt.show()
plt.close()

#!/usr/bin/env python3
import sys
sys.path.append('..')
import os
import numpy as np
import matplotlib.pyplot as plt
from methods import models
from methods import protocols
import seaborn as sns
colour_palette_1 = sns.dark_palette("#69d", 6, reverse=True)
colour_palette_2 = sns.dark_palette("seagreen", 6, reverse=True)

def main(model, data_file, g, meta):
    # Loop through different alphas to see the effects.
    x = models.VCModel(model, False, False, models.VC_FULL)
    y = models.VCModel(model, False, False, models.VC_IDEAL)
    p = [
        -80, 10,
        -20, 10,
    ]
    x.set_protocol(protocols.from_steps(p), dt=0.01)
    y.set_protocol(protocols.from_steps(p), dt=0.01)

    data = np.loadtxt('data/' + data_file + '.csv', delimiter=',')
    w = int(2640/0.1), int(2660/0.1) #2649, 2655
    data = data[w[0]:w[1], :]
    data[:, 0] -= data[0, 0]
    cm = meta[0]  # pF
    rs = meta[1]  # GOhm

    #print(x.fit_parameter_names())

    param = [g]
    #print(param)

    x.set_artefact_parameters({
        'cell.Cm': cm,
        'voltage_clamp.R_series': rs,
        'voltage_clamp.C_prs': 0,
        'voltage_clamp.V_offset_eff': 0,
        'voltage_clamp.Cm_est': cm,
        'voltage_clamp.R_series_est': rs,
        'voltage_clamp.C_prs_est': 0,
        #'voltage_clamp.alpha_R': 0.7,
        #'voltage_clamp.alpha_P': 0.7,
        'voltage_clamp.g_leak': 0.,
        'voltage_clamp.g_leak_est': 0.,
    })


    fig, axes = plt.subplots(3, 2, sharex=True, sharey='row', figsize=(12, 6.5), gridspec_kw={'height_ratios': [1, 2, 2]})

    axes[0, 0].plot(y.times(), y.voltage(param), c='#7f7f7f', ls='--')
    axes[0, 1].plot(y.times(), y.voltage(param), c='#7f7f7f', ls='--')
    axes[1, 0].plot(y.times(), y.simulate(param), c='#7f7f7f', ls='--', label='ideal')
    axes[1, 1].plot(y.times(), y.simulate(param), c='#7f7f7f', ls='--', label='ideal')

    alphas = [(0, 0), (0, 20), (0, 40), (0, 60), (0, 80)]
    for i, (r, p) in enumerate(alphas):
        x.set_artefact_parameters({
            'voltage_clamp.alpha_R': r / 100.,
            'voltage_clamp.alpha_P': p / 100.,
        })
        axes[0, 0].plot(x.times(), x.voltage(param), c=colour_palette_1[i])
        axes[1, 0].plot(x.times(), x.simulate(param), c=colour_palette_1[i], label=r'$\alpha_R = %d, \alpha_P = %d$' % (r, p))
        axes[2, 0].plot(data[:, 0], data[:, i+1] * cm, c=colour_palette_1[i])
    axes[1, 0].legend(loc=4)

    alphas = [(0, 70), (20, 70), (40, 70), (60, 70), (80, 70)]
    for i, (r, p) in enumerate(alphas):
        x.set_artefact_parameters({
            'voltage_clamp.alpha_R': r / 100.,
            'voltage_clamp.alpha_P': p / 100.,
        })
        axes[0, 1].plot(x.times(), x.voltage(param), c=colour_palette_2[i])
        axes[1, 1].plot(x.times(), x.simulate(param), c=colour_palette_2[i], label=r'$\alpha_R = %d, \alpha_P = %d$' % (r, p))
        axes[2, 1].plot(data[:, 0], data[:, i+1+5] * cm, c=colour_palette_2[i])
    axes[1, 1].legend(loc=4)

    axes[0, 0].set_xlim([9, 15])
    axes[1, 0].set_ylim([-4760, 950])
    axes[2, 0].set_ylim([-4760, 950])

    axes[0, 0].set_ylabel('Voltage (mV)')
    axes[1, 0].set_ylabel('Simulated current (pA)')
    axes[2, 0].set_ylabel('Experimental current (pA)')
    axes[2, 0].set_xlabel('Time (ms)')
    axes[2, 1].set_xlabel('Time (ms)')

    ax = axes[-1, 0]
    labels = [int(item.get_text()) - 9 for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)

    fig.tight_layout()
    plt.savefig(f'compensation-level-sweeps-2/compensation-level-sweeps-2-{os.path.basename(model.strip(".mmt"))}-{data_file}.pdf', format='pdf')
    plt.close()
    #plt.show()


model_list = ['paci-2020-ina.mmt', models.mmt('gray')]
gs_list = [[2 / 2.5, 5 / 2.5, 7 / 2.5, 8 / 2.5], [0.2, 0.5, 0.7, 0.8]]
data_list = ['4_021821_2_alex_control', '4_021821_4_alex_control', '4_021921_1_alex_control', '4_021921_2_alex_control']
meta_list = [(21.7, 44.2e-3), (27.4, 19.5e-3), (51.5, 25e-3), (51.0, 39.3e-3)]

for model, gs in zip(model_list, gs_list):
    for data, g, meta in zip(data_list, gs, meta_list):
        main(model, data, g, meta)

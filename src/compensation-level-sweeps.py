#!/usr/bin/env python3
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from methods import models
from methods import protocols
import seaborn as sns
colour_palette_1 = sns.dark_palette("#69d", 6, reverse=True)
colour_palette_2 = sns.dark_palette("salmon", 6, reverse=True)

# Loop through different alphas to see the effects.
x = models.VCModel('paci-2020-ina.mmt', False, False, models.VC_FULL)
y = models.VCModel('paci-2020-ina.mmt', False, False, models.VC_IDEAL)
#x = models.VCModel(models.mmt('kernik'), False, False, models.VC_MIN)
#y = models.VCModel(models.mmt('kernik'), False, False, models.VC_IDEAL)
p = [
    -80, 10,
    -20, 10,
]
x.set_protocol(protocols.from_steps(p), dt=0.01)
y.set_protocol(protocols.from_steps(p), dt=0.01)

data = np.loadtxt('data/4_021821_4_alex_control.csv', delimiter=',')
w = int(2640/0.1), int(2660/0.1) #2649, 2655
data = data[w[0]:w[1], :]
data[:, 0] -= data[0, 0]
cm = 27.4  # pF
rs = 19.5e-3  # GOhm

print(x.fit_parameter_names())

param = [2]#[5]#[3.5]
print(param)

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

shiftx = 10

axes[0, 1].plot(y.times() - shiftx, y.voltage(param), c='#7f7f7f', ls='--')
axes[0, 0].plot(y.times() - shiftx, y.voltage(param), c='#7f7f7f', ls='--')
axes[1, 1].plot(y.times() - shiftx, y.simulate(param), c='#7f7f7f', ls='--', label='ideal')
axes[2, 1].plot(y.times() - shiftx, y.simulate(param), c='#7f7f7f', ls='--', label='ideal')

alphas = [(0, 0), (0, 20), (0, 40), (0, 60), (0, 80)]
for i, (r, p) in enumerate(alphas):
    x.set_artefact_parameters({
        'voltage_clamp.alpha_R': r / 100.,
        'voltage_clamp.alpha_P': p / 100.,
    })
    axes[0, 1].plot(x.times() - shiftx, x.voltage(param), c=colour_palette_1[i])
    axes[1, 1].plot(x.times() - shiftx, x.simulate(param), c=colour_palette_1[i], label=r'$\alpha_R = %d, \alpha_P = %d$' % (r, p))
    axes[1, 0].plot(data[:, 0] - shiftx, data[:, i+1] * cm, c=colour_palette_1[i])
axes[1, 1].legend(loc=4)

alphas = [(0, 70), (20, 70), (40, 70), (60, 70), (80, 70)]
for i, (r, p) in enumerate(alphas):
    x.set_artefact_parameters({
        'voltage_clamp.alpha_R': r / 100.,
        'voltage_clamp.alpha_P': p / 100.,
    })
    axes[0, 1].plot(x.times() - shiftx, x.voltage(param), c=colour_palette_2[i])
    axes[2, 1].plot(x.times() - shiftx, x.simulate(param), c=colour_palette_2[i], label=r'$\alpha_R = %d, \alpha_P = %d$' % (r, p))
    axes[2, 0].plot(data[:, 0] - shiftx, data[:, i+1+5] * cm, c=colour_palette_2[i])
axes[2, 1].legend(loc=4)

axes[0, 1].set_xlim([9.1 - shiftx, 15 - shiftx])

axes[0, 0].set_ylabel('Voltage (mV)', fontsize=12)
axes[1, 0].set_ylabel('Current (pA)', fontsize=12)
axes[2, 0].set_ylabel('Current (pA)', fontsize=12)
axes[2, 0].set_xlabel('Time (ms)', fontsize=12)
axes[2, 1].set_xlabel('Time (ms)', fontsize=12)

axes[0, 1].set_title('Simulations')
axes[0, 0].set_title('Data')

for i in [0, 1]:
    axes[1, i].annotate(
        '', xy=(11 - shiftx, -2500), xytext=(12 - shiftx, -2500),
        arrowprops=dict(arrowstyle='->', lw=2, color=colour_palette_1[0],)
    )
    #axes[1, i].text(11.5, -2600, s=r'$\uparrow$$\alpha_P$',
    axes[1, i].text(11.5 - shiftx, -2700, s='Increasing\nprediction',
                    fontsize=12, ha='center', va='top')
    axes[2, i].annotate(
        '', xy=(11.5 - shiftx, -3500), xytext=(11.5 - shiftx, -2000),
        arrowprops=dict(arrowstyle='->', lw=2, color=colour_palette_2[0])
    )
    #axes[2, i].text(11.6, -2750, s=r'$\uparrow$$\alpha_R$',
    axes[2, i].text(11.6 - shiftx, -2750, s=r'Increasing $R_s$''\ncompensation',
                    fontsize=12, ha='left', va='center')

#ax = axes[-1, 1]
#labels = [int(item.get_text()) - 9 for item in ax.get_xticklabels()]
#ax.set_xticklabels(labels)

fig.align_ylabels(axes[:, 0])

fig.tight_layout()
plt.savefig('compensation-level-sweeps.pdf', format='pdf')
plt.show()

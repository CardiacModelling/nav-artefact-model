#!/usr/bin/env python3
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
from methods import models
from methods import protocols

x = models.VCModel(models.mmt('hh'), False, False, models.VC_FULL)
y = models.VCModel(models.mmt('hh'), False, False, models.VC_IDEAL)
p = [
    -100, 10,
    10, 20,
    -100, 10,
]
x.set_protocol(protocols.from_steps(p), dt=0.01)
y.set_protocol(protocols.from_steps(p), dt=0.01)

print(x.fit_parameter_names())

x.set_artefact_parameters({
    'cell.Cm': 10.,
    'voltage_clamp.R_series': 6e-3,
    'voltage_clamp.C_prs': 0,
    'voltage_clamp.V_offset_eff': 0,
    'voltage_clamp.Cm_est': 10.,
    'voltage_clamp.R_series_est': 6e-3,
    'voltage_clamp.C_prs_est': 0,
    #'voltage_clamp.alpha_R': 0.7,
    #'voltage_clamp.alpha_P': 0.7,
    'voltage_clamp.g_leak': 0.5,
    'voltage_clamp.g_leak_est': 0.4,
})


fig, axes = plt.subplots(2, 1, sharex=True)

axes[0].plot(y.times(), y.voltage([.1] * x.n_parameters()))
axes[1].plot(y.times(), y.simulate([.1] * y.n_parameters()), label='ideal')

for i in range(1, 9):
    x.set_artefact_parameters({
        'voltage_clamp.alpha_R': i / 10.,
        'voltage_clamp.alpha_P': i / 10.,
    })
    axes[0].plot(x.times(), x.voltage([.1] * x.n_parameters()))
    axes[1].plot(x.times(), x.simulate([.1] * x.n_parameters()), label=f'alpha={i/10.}')
axes[1].set_xlim([9.5, 12.5])
axes[1].legend()
plt.show()

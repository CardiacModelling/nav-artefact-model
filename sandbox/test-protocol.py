#!/usr/bin/env python3
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
from methods import models
from methods import protocols

x = models.VCModel(models.mmt('kernik'), True, True, models.VC_FULL)
y = models.VCModel(models.mmt('kernik'), True, True, models.VC_IDEAL)
p = protocols.load('../protocols/ina-steps.txt')
x.set_protocol(p, dt=0.04)
y.set_protocol(p, dt=0.04)

print(x.current_names())
print(x.n_parameters())
print(x.full_parameter_names())
print(x.fit_parameter_names())
print(y.n_parameters())
print(y.full_parameter_names())
print(y.fit_parameter_names())

x.generate_artefact_parameters()

plt.plot(x.times(), x.simulate([1] * x.n_parameters()), label='realistic')
plt.plot(y.times(), y.simulate([1] * y.n_parameters()), label='ideal')
plt.legend()
plt.show()

tx, cx = protocols.fold(x.times(), x.simulate([1] * x.n_parameters()), 40)
ty, cy = protocols.fold(y.times(), y.simulate([1] * y.n_parameters()), 40)
assert(len(cx) == 13)
assert(len(cy) == 13)

plt.subplot(211)
for i in cx.keys():
    plt.plot(tx, cx[i])
plt.ylabel('Realistic current')
plt.subplot(212)
for i in cy.keys():
    plt.plot(ty, cy[i])
plt.ylabel('Ideal current')
plt.show()

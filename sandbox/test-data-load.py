#!/usr/bin/env python3
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import methods.data

_, axes = plt.subplots(2, 1, sharex=True)
t, v, c = methods.data.load_named('cell1', pname='NaIVCP80')
for vi in v.keys():
    axes[0].plot(t, v[vi])
    axes[1].plot(t, c[vi])
axes[-1].set_xlabel('Time (ms)')
axes[0].set_ylabel('Voltage (mV)')
axes[1].set_ylabel('Current (pA)')

plt.show()


_, axes = plt.subplots(2, 1, sharex=True)
for cp in range(0, 90, 10):
    t, v, c = methods.data.load_named('cell2', pname=f'NaIVCP{cp}')
    vi = 10
    axes[0].plot(t, v[vi])
    axes[1].plot(t, c[vi])
axes[1].set_xlim([9.5, 12.5])
axes[-1].set_xlabel('Time (ms)')
axes[0].set_ylabel('Voltage (mV)')
axes[1].set_ylabel('Current (pA)')

plt.show()

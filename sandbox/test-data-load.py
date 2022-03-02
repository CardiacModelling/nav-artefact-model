#!/usr/bin/env python3
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import methods.data

t, v, c = methods.data.load_named('real', cell='0', pname='NaIVCP80')

_, axes = plt.subplots(2, 1, sharex=True)
for vi in v.keys():
    axes[0].plot(t, v[vi])
    axes[1].plot(t, c[vi])
axes[-1].set_xlabel('Time (ms)')
axes[0].set_ylabel('Voltage (mV)')
axes[1].set_ylabel('Current (A/F)')

plt.show()

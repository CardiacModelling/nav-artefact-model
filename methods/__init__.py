#
# Methods module
#

# Get methods directory
import os, inspect
frame = inspect.currentframe()
DIR_METHOD = os.path.abspath(os.path.dirname(inspect.getfile(frame)))
del(os, inspect, frame)

#
# Settings
#
run = 1

# Required currents and conductances (and parameter order)
required_currents = [
    '_Na',
]
optional_currents = [
]

concentrations = {
    'Na_i': 10,
    'Na_o': 140,
    'K_i': 130,  # 0?
    'K_o': 4,
    'Ca_i': 1e-5,
    'Ca_o': 2,
}

if run == 1:
    # Voltage, concentrations clamped
    results = 'results'

    t_hold = 2000  # ms
    v_hold = -100  # mV

else:
    raise ValueError('Unknown run number')

# Make results dir
import os
if not os.path.isdir(results):
    os.makedirs(results)
del(os)

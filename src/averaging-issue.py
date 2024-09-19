#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

normalise = '--norm' in sys.argv

all_v = list(range(-90, 50, 5))
#all_params = pd.read_csv('./data/mod_simulations-fig2/all_params.csv')

#all_currs = pd.read_csv('./data/mod_simulations-fig2/all_currs.csv')
all_iv = pd.read_csv('./data/mod_simulations-fig2/all_iv.csv')

#baseline_curr = pd.read_csv('./data/mod_simulations-fig2/baseline_curr.csv')
baseline_dat = np.loadtxt('./data/mod_simulations-fig2/baseline.csv')

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
#fig.subplots_adjust(0.23, 0.2, 0.99, 0.99)

sem = lambda x: np.std(x, axis=0) / np.sqrt(len(x))

np.random.seed(0)
choice1 = np.random.choice(len(all_iv.T), 5, replace=False)
choice2 = np.random.choice(len(all_iv.T), 25, replace=False)
choice3 = np.random.choice(len(all_iv.T), len(all_iv.T), replace=False)

alphas = [0.25, 0.15, 0.1]

# Simulations
for ax, choice, alpha in zip(axes, [choice1, choice2, choice3], alphas):
    normalised = []
    for iv in all_iv.values[:, choice].T:
        if normalise:
            normalised.append(iv / np.max(np.abs(iv)))
        else:
            normalised.append(iv)
        ax.plot(all_v, normalised[-1], 'grey', alpha=alpha)

    m = np.mean(normalised, axis=0)
    line1 = ax.plot(all_v, m, 'C0', marker='o', mfc='none', label='Average')
    ebars = ax.errorbar(all_v, m, yerr=sem(normalised), c='C0')

    i = np.argmin(m)
    print('Average:', linregress(all_v[i+1:i+7], m[i+1:i+7]))

    if normalise:
        b = baseline_dat / np.max(np.abs(baseline_dat))
    else:
        b = baseline_dat
    line2 = ax.plot(all_v, b, c=(.8, .1, .1), marker='o', label='Physiological')

    i = np.argmin(b)
    print('Physiological:', linregress(all_v[i+1:i+7], b[i+1:i+7]))

    ax.set_xlabel('Voltage (mV)', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('n = {}'.format(len(normalised)), fontsize=10, loc='right')

for i in [1, 2]:
    axes[i].axes.get_yaxis().set_visible(False)
    axes[i].spines['left'].set_visible(False)

axes[0].set_ylabel('Current density (pA/pF)', fontsize=12)
axes[-1].legend(loc=4, fontsize=10)
axes[-1].set_xlim(-96, 59)

plt.tight_layout()
postfix = '-norm' if normalise else ''
plt.savefig(f'./averaging-issue{postfix}.pdf', format='pdf')

plt.show()

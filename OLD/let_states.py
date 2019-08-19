"""
Simulate N(t) = ceiling(lambda * t), with lambda normally distributed.
In "Learning to time: a perspective", Appendix, they simulate two 'walks'
with lambda = {0.8,1.2} sampled from N(1,0.2)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def n_t(t, mean, sd):
    """
    lambda admit just positive values
    """

    lmbd = np.random.normal(mean, sd, 1)
    vec_time = np.arange(t)
    nt_iter = np.zeros(t)

    while lmbd < 0:
        lmbd = np.random.normal(mean, sd, 1)

    for j in vec_time:
        nt_iter[j] = np.ceil(lmbd*j)
    return nt_iter


T = 40
time = np.arange(T) + 1
m = 1
std = 0.2
trials = 200
trial_rf = np.zeros(trials)

gs = gridspec.GridSpec(1, 4, wspace=0.06)
ax1 = plt.subplot(gs[0, 0:3])
ax2 = plt.subplot(gs[0, 3])

for jj in np.arange(trials):
    Nt = n_t(T, m, std)
    trial_rf[jj] = Nt[-1]
    ax1.step(time, Nt, c='grey', lw=1, alpha=0.3)
    ax1.scatter(time[-1], Nt[-1], c='black', s=10, alpha=0.5)

ax1.text(10, np.max(trial_rf) - 5,
         r'$\lambda \sim \mathcal{{N}}(\mu = 1, \sigma = {{{}}})$'.format(std) + "\n" +
         r'$T = {{{}}}$'.format(T) + ", {} trials".format(trials),
         {'color': 'k', 'fontsize': 10, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
ax1.set_xlabel('$t$ (time in trial)')
ax1.set_ylabel(r'$N(t) = \lceil \lambda t \rceil$')
ax2.hist(trial_rf, orientation="horizontal", histtype='step', linewidth=0.8,
         facecolor='grey', edgecolor='k', fill=True, alpha=0.5)
ax2.yaxis.tick_right()
ax2.set_ylim(0, np.max(trial_rf))
ax2.yaxis.set_label_position("right")
ax2.axhline(y=np.mean(trial_rf), color='black', linestyle=":")
ax2.set_ylabel(r'$N(t = T)$')
# plt.savefig('nt_let.pdf', dpi = 120)
plt.show()

"""
Simulations Learning to Time

    lambda ~ N(x, mu, sigma), and lambda > 0; is the transition rate between states
    every j trial a new value of lambda is sampled.
    N(t) is the state active in the t-time since trial, which depends on lambda and t,
    so, N(t) = ceil (lambda * t)

"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
rc('text', usetex=True)  # if LaTeX is installed, otherwise comment
rc('figure', figsize=(5.5, 4))
rc('axes', ymargin=0)

"""
Simulate N(t) = ceiling(lambda * t), with lambda normally distributed. In "Learning to time: a perspective", 
Appendix, they simulate two 'walks' with lambda = {0.8,1.2} sampled from N(1,0.2)
"""

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


t = 40
time = np.arange(t)+1
mean = 1
sd = 0.2
trials = 200
trial_rf = np.zeros(trials)

gs = gridspec.GridSpec(1, 4, wspace = 0.06)
ax1 = plt.subplot(gs[0,0:3])
ax2 = plt.subplot(gs[0,3])

for jj in np.arange(trials):
    Nt = n_t(t, mean, sd)
    trial_rf[jj] = Nt[-1]
    ax1.step(time,Nt, c='grey', lw=1, alpha=0.3)
    ax1.scatter(time[-1], Nt[-1], c='black', s=10, alpha=0.5)

ax1.text(10, np.max(trial_rf) - 5,
         r'$\lambda \sim \mathcal{{N}}(\mu = 1, \sigma = {{{}}})$'.format(sd) + "\n" +
         r'$T = {{{}}}$'.format(t) + ", {} trials".format(trials),
         {'color': 'k', 'fontsize': 10, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
ax1.set_xlabel('$t$ (time in trial)')
ax1.set_ylabel(r'$N(t) = \lceil \lambda t \rceil$')
ax2.hist(trial_rf, orientation="horizontal", histtype='step', linewidth = 0.8,
         facecolor = 'grey',edgecolor='k', fill = True, alpha = 0.5)
ax2.yaxis.tick_right()
ax2.set_ylim(0,np.max(trial_rf))
ax2.yaxis.set_label_position("right")
ax2.axhline(y = np.mean(trial_rf), color = 'black', linestyle = ":")
ax2.set_ylabel(r'$N(t = T)$')
# plt.savefig('nt_let.pdf', dpi = 120)
plt.show()

"""
Old sim, with gamma distribution. 
def learn2time(t, T, parameters):

    lambd = parameters[0]
    gamma = parameters[1]
    A = parameters[2]
    R0 = parameters[3]
    T2 = T * 3  # 3x time of ref
    p = 0.85

    jx = 1
    X = np.zeros((len(t), 80))
    r = np.zeros((len(t), 80))
    for j in range(80):
        jx = jx * (j + 1)
        X[t, j] = (1 / jx) * np.exp(-lambd * t) * (lambd * t) ** (j + 1)
        Xt = np.exp(-lambd * T) * (lambd * T) ** (j + 1) / jx
        Xr1 = 0
        for i in range(T):
            Xs = np.exp(- lambd * i) * (lambd * i) ** (j + 1) / jx
            Xr1 = Xr1 + Xs
        Xr1 = 1 * Xr1

        Xr2 = 0
        for i in range(T2):
            Xs = np.exp(- lambd * i) * (lambd * i) ** (j + 1) / jx
            Xr2 = Xr2 + Xs
        Xr2 = 1 * Xr2
        W = Xt / (Xt + gamma * Xr2 + gamma * (1 - p) / (p * Xr2))
        r[t, j] = W * X[t, j]
    R = A * np.sum(r, axis=1) * R0
    return R


T = 30
t = np.arange(0, T * 3)
parameters = np.array([0.106, 0.017, 94.734, 3.072])

Rsim = learn2time(t, T, parameters)

fig = plt.figure(figsize=(6, 6), dpi=90)
plt.plot(t, Rsim, linewidth=0.5, color='grey')
plt.title('Learning to Time')
plt.axvline(x=t[np.argmax(Rsim)], color='red', linestyle='--')
plt.text(t[np.argmax(Rsim)] + 5, np.max(Rsim), 'time is %d sec' % t[np.argmax(Rsim)])
plt.show()
"""

"""
How to compute probabilities:

Example of P(lambda > n/T), if n = 15, T = 40. Intuitively P will be high
First let see how this density looks with the following code. 
Copy-paste and run

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)  # if LaTeX is installed, otherwise comment
rc('figure', figsize=(5.5, 4))

T, n = 40, 35

t = np.arange(0, 100, 0.1)
pt = norm.pdf(t, loc=1 * T, scale=0.2 * T)
pla = 1 - norm.cdf(n/T,loc=1,scale=0.2).round(5)
# rc('axes', ymargin=0.01)

fig, ax = plt.subplots()
ax.plot(t, pt, 'b-', linewidth=1)
ax.axvline(x=n, linestyle="--", color = 'b',lw = 1)
ax.fill_between(t[t>n],pt[t>n],0, alpha=0.5, color='grey')
ax.annotate("$n = {}$".format(n),xy = (n,pt[t == n]), xytext = (0,0.01),
            arrowprops=dict(arrowstyle="->"))
ax.text(60, np.max(pt)/2,
         r'$P(\lambda > \frac{{35}}{{40}}) \approx {{{}}}$'.format(pla),
         {'color': 'k', 'fontsize': 10, 'ha': 'center', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.5)})
ax.set(xlabel='$t$',ylabel="density")
plt.savefig('lambda_nT.pdf', dpi = 120)
plt.show()



The area AFTER the dashed line is the probability we are looking for. You can see is very large. 
That can be computed with

1 - norm.cdf(n/T,loc=1*T,scale=0.2*T)

They provide an approximation for Phi(n/T,mu,sigma) - Phi((n-1)/T,mu,sigma),
which I think is not necessary, since sofware have cdf

n = 44
T = 45
mu = 1
sig = 0.2


print(norm.cdf(n/T,mu,sig) - norm.cdf((n-1)/T,mu,sig))

print(1/T * norm.pdf(n/T,mu,sig))

"""

"""
Which is the most probable state being reinforced?
Likelihood function

T, ns = 40, 35
rc('text', usetex=True)
n = np.arange(0, T * 3, 1)
lik = norm.pdf(n, loc=1 * T, scale=0.2 * T)
ml_state = n[np.argmax(lik)]
bel_state = lik[n == ns]
fig, ax = plt.subplots()
ax.plot(n, lik, 'b-', linewidth=1)
ax.scatter(ml_state, np.max(lik))
ax.annotate(r"Max $\mathcal L (n) :{}$".format(ml_state), xy=(ml_state, np.max(lik)),
            xytext=(T * 1.2, np.max(lik) / 1.2),
            arrowprops=dict(arrowstyle="->"))
ax.scatter(ns, bel_state)
ax.annotate(r"$\mathcal L (n<T) :{}$".format(ns), xy=(ns, bel_state),
            xytext=(ns, bel_state / 1.2),
            arrowprops=dict(arrowstyle="->"))
ax.set(xlabel = 'states ($n$)')
ax.set(ylabel='$p(n = T)$')
plt.show()


"""

"""
tests: how much is epsilon in the ceiling function?
def epsi(t, mean, sd):
    lmbd = np.random.normal(mean, sd, 1)
    vec_time = np.arange(t)
    nt_iter = np.zeros(t)

    while lmbd < 0:
        lmbd = np.random.normal(mean, sd, 1)

    for j in vec_time:
        nt_iter[j] = np.ceil(lmbd * j) - lmbd * j
    return nt_iter.mean()


import numpy as np
import matplotlib.pyplot as plt

t = 40
mean = 1
sd = 0.01
trials = 200
eps = np.zeros(trials)

for jj in np.arange(trials):
    eps[jj] = epsi(t, mean*t, sd*t)

plt.hist(eps)
plt.axvline(x = np.mean(eps), color='r')
plt.show()




    
"""

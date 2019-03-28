"""
Simulations Learning to Time

    lambda ~ N(x, mu, sigma), and lambda > 0; is the transition rate between states
    every j trial a new value of lambda is sampled.
    N(t) is the state active in the t-time since trial, wich depends on lambda and t,
    so, N(t) = ceil (lambda * t)


"""
import numpy as np
import matplotlib.pyplot as plt


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
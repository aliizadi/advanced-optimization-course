import numpy as np
import math
import matplotlib.pyplot as plt


def f(x):
    x1 = x[0]
    x2 = x[1]
    return (x2 - 0.129 * (x1 ** 2) + 1.6 * x1 - 6) ** 2 + 6.07 * math.cos(x1) + 10


def grad_f(x):
    x1 = x[0]
    x2 = x[1]

    return np.array([2 * (x2 - 0.129 * (x1 ** 2) + 1.6 * x1 - 6) * (-0.129 * 2 * x1 + 1.6) + 6.07 * math.sin(x1),
                     2 * (x2 - 0.129 * (x1 ** 2) + 1.6 * x1 - 6)])


def hessian_f(x):
    x1 = x[0]
    x2 = x[1]

    return np.array([[2 * (2 * 3 * (0.129) ** 2 * (
        x1) ** 2 - 1.6 * 2 * 0.129 * x1 - 2 * 2 * 1.6 * 0.129 * x1 + 1.6 ** 2) - 6.07 * math.cos(x1),
                      2 * (-2 * 0.129 * x1 + 1.6)],
                     [2 * (-2 * 0.129 * x1 + 1.6),
                      2]])


x0 = np.array([6, 14])

delta_0 = 2
delta_hat = 5


def cauchy_point(x, B, delta):
    g = grad_f(x)
    gT_B_g = np.dot(np.dot(g, B), g)
    g_norm = np.linalg.norm(g)

    if gT_B_g <= 0:
        taw = 1
    else:
        taw = min(g_norm ** 3. / (delta * gT_B_g), 1.)
    return -1. * (taw * delta / g_norm) * g


def dogleg(H, g, B, delta):
    pb = -H @ g

    if np.linalg.norm(pb) <= delta:
        return pb

    pu = - (np.dot(g, g) / np.dot(g, B @ g)) * g
    dot_pu = np.dot(pu, pu)
    norm_pu = np.sqrt(dot_pu)
    if norm_pu >= delta:
        return delta * pu / norm_pu

    # ||pu**2 +(tau-1)*(pb-pu)**2|| = delta**2
    pb_pu = pb - pu
    pb_pu_sq = np.dot(pb_pu, pb_pu)
    pu_pb_pu_sq = np.dot(pu, pb_pu)
    d = pu_pb_pu_sq ** 2 - pb_pu_sq * (dot_pu - delta ** 2)
    tau = (-pu_pb_pu_sq + np.sqrt(d)) / pb_pu_sq + 1
    if tau < 1:
        return pu * tau
    return pu + (tau - 1) * pb_pu


def trust_region(x0, eta=0.2, delta_0=2, delta_hat=5, p_func='dogleg'):
    xs = []
    fs = []
    x = x0
    delta = delta_0
    xs.append(x)
    for iteration in range(20):
        g = grad_f(x)
        B = hessian_f(x)

        H = np.linalg.inv(B)

        if p_func == 'dogleg':
            p = dogleg(H, g, B, delta)
        elif p_func == 'cauchy':
            p = cauchy_point(x, B, delta)

        rho = (f(x) - f(x + p)) / (-(g @ p + 0.5 * p.T @ B @ p))

        if rho < 0.25:
            delta = 0.25 * delta
        else:
            if rho > 0.75 and np.linalg.norm(p) == delta:
                delta = min(2.0 * delta, delta_hat)
            else:
                delta = delta

        if rho > eta:
            x = x + p
        xs.append(x)

        if np.linalg.norm(g) < 0.0001:
            break

        fs.append(f(x))

    return xs, fs


xs, fs = trust_region(x0)

print(fs[-1])
print(xs[-1])

plt.plot(fs)

plt.figure()
print()

xs, fs = trust_region(x0, p_func='cauchy')

print(fs[-1])
print(xs[-1])
plt.plot(fs)


plt.show()

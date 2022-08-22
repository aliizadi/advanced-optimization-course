import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def f(x):
    x1 = deepcopy(x[0])
    x2 = deepcopy(x[1])
    x3 = deepcopy(x[2])
    return x1 ** 2 + 2 * x2 ** 2 + 3 * x3 ** 2


def grad_f(x):
    x1 = deepcopy(x[0])
    x2 = deepcopy(x[1])
    x3 = deepcopy(x[2])
    return np.array([2 * x1, 4 * x2, 6 * x3])


def hessian_f(x):
    return np.array([[2, 0, 0], [0, 4, 0], [0, 0, 6]])


def I_plus(x):
    g = grad_f(x)
    indices = []
    for i, (x_i, g_i) in enumerate(zip(x, g)):
        if x_i == 0 and g_i > 0:
            indices.append(i)

    return indices


def modified_D(x, D):
    I_plus_indices = I_plus(x)
    for i in I_plus_indices:
        for j in range(D.shape[1]):
            if j != i:
                D[i][j] = 0
    return D


def c_c_projection(x):
    projected = np.zeros(3)
    for i in range(3):
        if x[i] < 0:
            projected[i] = 0
        else:
            projected[i] = x[i]
    return projected


def backtracking_line_search(x, p, initial_alpha=1, c1=0.9, rho=0.5):
    def armijo(alpha):
        return f(x + alpha * p) <= f(x) + c1 * alpha * np.dot(grad_f(x), p)

    alpha = initial_alpha
    while not armijo(alpha):
        alpha = alpha * rho
    return alpha


x0 = np.array([0.0, 0.0, 1.0])


def run_two_metric_projection():
    alpha = 1
    x = x0
    fs = []
    fs.append(f(x))
    for iteration in range(100):
        g = grad_f(x)
        H = hessian_f(x)
        D = np.linalg.inv(H)
        D = modified_D(x, D)
        alpha = backtracking_line_search(x, -D @ g, alpha)
        x = x - alpha * np.dot(D, g)
        x = c_c_projection(x)
        fs.append(f(x))

    return fs, x


f_xs, x_min = run_two_metric_projection()

print('x min: ', x_min)
plt.plot(f_xs)
plt.show()

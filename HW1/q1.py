import numpy as np
import matplotlib.pyplot as plt


# optimization

# define function, f(x1, x2) = 100 * (x2 - x1^2)^2 + (1 − x1)^2
def f(x1, x2):
    return 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


# Compute the gradient of f(x1, x2)
def grad_f(x1, x2):
    return np.array([
        400 * x1 ** 3 - 400 * x1 * x2 + 2 * x1 - 2,
        200 * x2 - 200 * x1 ** 2
    ])


# compute the Hessian of f(x1, x2)
def hessian_f(x1, x2):
    return np.array([
        [1200 * x1 ** 2 - 400 * x2 + 2, -400 * x1],
        [-400 * x1, 200]
    ])


def backtracking_line_search(x, p, initial_alpha=1, c1=10e-4, c2=0.99, rho=0.5):
    def armijo(alpha):
        return f(x[0] + alpha * p[0], x[1] + alpha * p[1]) <= f(x[0], x[1]) + \
               c1 * alpha * np.dot(grad_f(x[0], x[1]), p)

    def curvature(alpha):
        return np.dot(grad_f(x[0] + alpha * p[0], x[1] + alpha * p[1]), p) >= \
               c2 * np.dot(grad_f(x[0], x[1]), p)

    alpha = initial_alpha

    while not armijo(alpha):
        alpha = alpha * rho

    return alpha


method = '3'


def initialize_alpha(i, last_alpha, last_f, last_gradient, last_p, p_k, x):
    initial_alpha = 1

    if i >= 1:
        if method == '0':
            initial_alpha = 1

        if method == '1':
            initial_alpha = (2.0 * (f(x[0], x[1]) - last_f)) / np.dot(last_gradient, last_p)

        if method == '2':
            initial_alpha = (last_alpha * np.dot(last_gradient, last_p)) / np.dot(grad_f(x[0], x[1]), p_k)

        if method == '3':
            initial_alpha = (2.0 * (f(x[0], x[1]) - last_f)) / np.dot(grad_f(x[0], x[1]), p_k)
    return initial_alpha


class SteepestDescent:
    def __init__(self, x0, max_iter=1000):
        self.x0 = x0
        self.max_iter = max_iter

    def optimize(self):
        fs = []
        alphas = []

        x = self.x0

        last_alpha = 1
        last_gradient = grad_f(x[0], x[1])
        last_p = -grad_f(x[0], x[1])
        last_f = f(x[0], x[1])

        for i in range(self.max_iter):
            p_k = - grad_f(x[0], x[1])

            # compute alpha
            initial_alpha = initialize_alpha(i, last_alpha, last_f, last_gradient, last_p, p_k, x)
            alpha = backtracking_line_search(x, p_k, initial_alpha)

            last_x = x.copy()
            last_alpha = initial_alpha
            last_gradient = grad_f(x[0], x[1])
            last_p = -grad_f(x[0], x[1])
            last_f = f(x[0], x[1])

            # update x
            x = x + alpha * p_k

            fs.append(f(x[0], x[1]))
            alphas.append(alpha)

            if np.linalg.norm(x - last_x) < 1e-6:
                print(50 * '*', 'break')
                break
            print(alpha, f(x[0], x[1]), x[0], x[1])

        return x, fs, alphas


class Newton:
    def __init__(self, x0, max_iter=1000):
        self.x0 = x0
        self.max_iter = max_iter

    def optimize(self):
        fs = []
        alphas = []

        x = self.x0

        last_alpha = 1
        last_gradient = grad_f(x[0], x[1])
        last_p = -grad_f(x[0], x[1])
        last_f = f(x[0], x[1])

        for i in range(self.max_iter):
            p_k = - np.dot(np.linalg.inv(hessian_f(x[0], x[1])), grad_f(x[0], x[1]))

            # compute alpha
            initial_alpha = initialize_alpha(i, last_alpha, last_f, last_gradient, last_p, p_k, x)
            alpha = backtracking_line_search(x, p_k, initial_alpha)

            last_x = x.copy()
            last_alpha = initial_alpha
            last_gradient = grad_f(x[0], x[1])
            last_p = -grad_f(x[0], x[1])
            last_f = f(x[0], x[1])

            # update x
            x = x + alpha * p_k

            fs.append(f(x[0], x[1]))
            alphas.append(alpha)

            if np.linalg.norm(x - last_x) < 1e-6:
                print(50 * '*', 'break')
                break
            print(alpha, f(x[0], x[1]), x[0], x[1])
        return x, fs, alphas


x0 = np.array([-4, 10])

x, fs, alphas = SteepestDescent(x0).optimize()
# x, fs, alphas = Newton(x0).optimize()

plt.plot(fs)
plt.ylabel('f(x)')
plt.xlabel('iteration')
plt.annotate('f(x) = %.2f' % fs[-1], xy=(len(fs) - 5, 10 + fs[-1]))
plt.title(f'Steepest Descent - initial_step_length: method {method}')
plt.figure()

plt.ylabel('step length')
plt.xlabel('iteration')
plt.title(f'Steepest Descent - initial_step_length: method {method}')
plt.plot(alphas)
plt.show()

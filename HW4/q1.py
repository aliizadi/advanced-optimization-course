import numpy as np
import matplotlib.pyplot as plt


def f(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    return np.array([x1 ** 2 + 2 * x2 ** 2 + 3 * x3 ** 2])


def grad_f(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    return np.array([2 * x1, 4 * x2, 6 * x3])


def hessian_f(x):
    return np.array([[2, 0, 0], [0, 4, 0], [0, 0, 6]])


def constraint(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    return x1 + x2 + x3 - 1


def Q_penalty(x, mu):
    f(x) + (mu / 2.0) * constraint(x) ** 2


def grad_Q_penalty(x, mu):
    return grad_f(x) + mu * constraint(x) * np.array([1, 1, 1])


def hessian_Q_penalty(x, mu):
    return hessian_f(x) + mu * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def backtracking_line_search(x, p, initial_alpha=1, c1=0.9, rho=0.5):
    def armijo(alpha):
        return f(x + alpha * p) <= f(x) + c1 * alpha * np.dot(grad_f(x), p)

    alpha = initial_alpha
    while not armijo(alpha):
        alpha = alpha * rho
    return alpha


def quadratic_penalty_method():
    x0 = np.array([0.1, 0.2, 0.7])
    x = x0

    epsilon = 0.001
    mu = 1

    xs = []
    fs = []

    for i in range(100):
        if i % 10 == 0:
            print(i, mu)
        for j in range(10000):
            g_Q = grad_Q_penalty(x, mu)
            H_Q = hessian_Q_penalty(x, mu)
            x = x - 0.01 * np.dot(np.linalg.inv(H_Q), g_Q)

            if np.linalg.norm(g_Q) < epsilon:
                print('sub-problem converged')
                break

        mu = mu * 1.2

        xs.append(x)
        fs.append(f(x))

    return xs, fs


result_xs1, result_fs1 = quadratic_penalty_method()
#
#
# print(result_xs1[-1])
#
# plt.plot(result_fs1)
# plt.title("Quadratic Penalty Method")
# plt.xlabel("Iteration")
# plt.ylabel("f(x)")
#
# plt.figure()
# plt.plot(result_xs1)
# plt.title("Quadratic Penalty Method")
# plt.xlabel("Iteration")
# plt.ylabel("x")


def Q_multiplier(x, lambda_, mu):
    return f(x) + lambda_ * constraint(x) + (mu / 2.0) * constraint(x) ** 2


def grad_Q_multiplier(x, lambda_, mu):
    return grad_f(x) + lambda_ * np.array([1, 1, 1]) + mu * constraint(x) * np.array([1, 1, 1])


def hessian_Q_multiplier(x, lambda_, mu):
    return hessian_f(x) + mu * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def multiplier_method():
    x0 = np.array([0.1, 0.2, 0.7])
    x = x0

    epsilon = 0.001
    mu = 1
    lambda_ = 1

    xs = []
    fs = []

    for i in range(100):
        if i % 10 == 0:
            print(i, mu)
        for j in range(10000):
            g_Q = grad_Q_multiplier(x, lambda_, mu)
            H_Q = hessian_Q_multiplier(x, lambda_, mu)
            p = - np.dot(np.linalg.inv(H_Q), g_Q)
            x = x - 0.01 * -p

            if np.linalg.norm(g_Q) < epsilon:
                print('sub-problem converged')
                break

        mu = mu * 1.1
        lambda_ = lambda_ + mu * constraint(x)

        xs.append(x)
        fs.append(f(x))

    return xs, fs


result_xs2, result_fs2 = multiplier_method()
print(result_xs1[-1])
print(result_xs2[-1])

plt.plot(result_fs1, label='Penalty')
plt.plot(result_fs2, label='Multiplier')
plt.xlabel("Iteration")
plt.ylabel("f(x)")
plt.legend()

# plt.figure()
# # plt.plot(result_xs1, label='Penalty')
# plt.plot(result_xs2, label='Multiplier')
# plt.xlabel("Iteration")
# plt.ylabel("x")
# plt.legend()

plt.show()

import matplotlib.pyplot as plt
import numpy as np


class ConjugateGradient:
    def __init__(self, A, b, x0, max_iter=200000, epsilon=1e-6):
        self.A = A
        self.b = b
        self.x0 = x0
        self.max_iter = max_iter
        self.epsilon = epsilon

    def solve(self):

        rs = []

        A = self.A
        b = self.b

        x = self.x0

        r = A @ x - b
        p = - r

        rs.append(np.linalg.norm(r))

        for i in range(self.max_iter):

            alpha = - (r.T @ p) / (p.T @ (A @ p))
            x = x + alpha * p
            r = A @ x - b
            B = (r.T @ A @ p) / (p.T @ A @ p)
            p = - r + B * p

            rs.append(np.linalg.norm(r))

            if np.linalg.norm(r) < self.epsilon:
                print("Converged at iteration: ", i + 1)
                break

        return x, rs


def condition_number(A):
    eigen_values = np.linalg.eigvals(A)
    cond = np.max(eigen_values) / np.min(eigen_values)
    return cond


def solve_linear_equation(n=5):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = 1 / ((i + 1) + (j + 1) - 1)

    A_condition_number = condition_number(A)

    b = np.ones(n).reshape(n, 1)

    x0 = np.zeros(n).reshape(n, 1)

    cg = ConjugateGradient(A, b, x0)
    x_star, rs = cg.solve()

    print("n: ", n)
    print('condition_number: ', round(A_condition_number, 2))
    print("iterations needed: ", len(rs) - 1)
    print('last r: ', round(rs[-1], 9))
    print(40 * '-')

    plt.plot(rs)
    plt.xlabel('iteration')
    plt.ylabel('r')
    plt.title(f'CG method for A_n*n, n={n}\n last r={round(rs[-1], 9)}')
    plt.show()

    return x_star, rs


ns = [5, 8, 12, 20]

if __name__ == '__main__':
    for n in ns:
        solve_linear_equation(n)

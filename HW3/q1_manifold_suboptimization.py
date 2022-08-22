import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class ManifoldSubOptimization:
    def __init__(self, x0, active='b'):
        self.x0 = x0
        self.Q = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        if active == 'a':
            self.A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            self.b = np.array([0, 0, 0, 1])
            self.active = self.active_a

        elif active == 'b':
            self.A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [-1, 0, 0]])
            self.b = np.array([0, 0, 0, 1, -0.5])
            self.active = self.active_b

    def f(self, x):
        f = np.dot(np.dot(x.T, self.Q), x)
        return f

    @staticmethod
    def grad_f(x):
        x1 = deepcopy(x[0])
        x2 = deepcopy(x[1])
        x3 = deepcopy(x[2])
        return np.array([2 * x1, 4 * x2, 6 * x3])

    @staticmethod
    def hessian_f(x):
        return np.array([[2, 0, 0], [0, 4, 0], [0, 0, 6]])

    def active_a(self, x):
        active_a = np.dot(self.A, x) - self.b
        return np.where(active_a == 0)[0]

    def active_b(self, x):
        active_b = np.dot(self.A, x) - self.b
        return np.where(active_b == 0)[0]

    def sub_problem_solution(self, x, active_set):
        """
        The sub optimization problem - equality constraints restricted to the active set - solving using kkt systems
        """

        # find lagrange multipliers of main problem restricted to active set
        A = self.A[active_set]
        b = self.b[active_set]

        kkt = np.block([[self.Q, -A.T], [A, np.zeros((len(active_set), len(active_set)))]])
        rhs_kkt = np.block([np.zeros((len(self.Q))), b])

        sol = np.linalg.solve(kkt, rhs_kkt)

        # x = sol[:len(self.Q)]
        l = sol[len(self.Q):]

        # find minimum of the sub problem
        H = self.hessian_f(x)
        g = self.grad_f(x)

        kkt = np.block([[H, -A.T], [A, np.zeros((len(active_set), len(active_set)))]])
        rhs_kkt = np.block([-g, np.zeros(len(b))])
        sol = np.linalg.solve(kkt, rhs_kkt)
        x = sol[:len(H)]
        # l = sol[len(H):]

        return x, l

    def run(self):
        x = self.x0

        fs = [self.f(x)]

        active_set = list(self.active(x))

        for iteration in range(10):

            p, l = self.sub_problem_solution(x, active_set)

            if np.all(p <= 0.000001):
                if np.all(l >= 0):
                    return fs

                else:
                    j = active_set[np.argmin(l)]
                    active_set.remove(j)
            else:
                not_active_set = [i for i in range(len(self.A)) if i not in active_set]
                has_constraint = [i for i in not_active_set if self.A[i, :].dot(p) < 0]
                blocking = [(self.b[i] - self.A[i, :].dot(x)) / self.A[i, :].dot(p) for i in has_constraint]
                alpha = min([1] + blocking)
                if alpha != 1:
                    active_set.append(has_constraint[np.argmin(blocking)])
                x = x + alpha * p

            fs.append(self.f(x))

            print(x)
        return fs


x0 = np.array([0, 0, 1])

opt = ManifoldSubOptimization(x0)
fs = opt.run()
print(fs[-1])
plt.plot(fs)
plt.show()

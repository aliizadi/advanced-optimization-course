import numpy as np
import matplotlib.pyplot as plt

from q1 import ConjugateGradient, condition_number

n = 50
A1 = np.diag(np.random.uniform(1, 10, n))

e1 = np.random.normal(2, 0.3, 15)
e2 = np.random.normal(5, 0.3, 15)
e3 = np.random.normal(8, 0.3, 20)

assert (e1 > 0).all() and (e2 > 0).all() and (e3 > 0).all()

A2 = np.diag(e1.tolist() + e2.tolist() + e3.tolist())

plt.scatter(np.diag(A1), ['A1' for _ in range(len(np.diag(A1)))])
plt.scatter(np.diag(A2), ['A2' for _ in range(len(np.diag(A2)))])
plt.xlabel('Eigenvalues')
plt.ylabel('Matrix')
plt.title('Eigenvalues (diagonal elements) of A1 and A2')
plt.show()

A1_condition_number = condition_number(A1)
A2_condition_number = condition_number(A2)

b = np.ones(n).reshape(n, 1)

x0 = np.zeros(n).reshape(n, 1)

cg = ConjugateGradient(A1, b, x0)
x_star, rs = cg.solve()

print("A1")
print('condition_number: ', round(A1_condition_number, 2))
print("iterations needed: ", len(rs) - 1)
print('last r: ', round(rs[-1], 9))
print(40 * '-')

plt.plot(rs)
plt.xlabel('iteration')
plt.ylabel('r')
plt.title(f'CG method for A1,\nlast r={round(rs[-1], 9)}')
plt.show()

cg = ConjugateGradient(A2, b, x0)
x_star, rs = cg.solve()

print("A2")
print('condition_number: ', round(A2_condition_number, 2))
print("iterations needed: ", len(rs) - 1)
print('last r: ', round(rs[-1], 9))
print(40 * '-')

plt.plot(rs)
plt.xlabel('iteration')
plt.ylabel('r')
plt.title(f'CG method for A2,\nlast r={round(rs[-1], 9)}')
plt.show()

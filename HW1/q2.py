import math
import numpy as np
import random
import matplotlib.pyplot as plt


def f(x):
    return x ** 3 - 60 * x ** 2 + 900 * x + 100


def simulated_annealing(T0, cooling_coef, x):
    fs = []
    x_value = int(x, 2)
    T = T0
    for i in range(100):
        x_new = x

        # change one bit randomly
        bit_i = np.random.randint(0, 5)
        x_new = list(x_new)
        x_new[bit_i] = str(1 - int(x_new[bit_i]))
        x_new = ''.join(x_new)
        x_new_value = int(x_new, 2)

        delta_E = f(x_new_value) - f(x_value)
        fs.append(f(x_value))

        # accept or reject
        if delta_E <= 0:
            x = x_new
            x_value = x_new_value
        else:
            # probability of accepting
            p = math.exp(-delta_E / T)
            if random.random() < p:
                x = x_new
                x_value = x_new_value

        # cooling
        T *= cooling_coef

    return x, fs


x = '10011'

T0 = 500
cooling_coef = 0.9
x, fs = simulated_annealing(T0, cooling_coef, x)

plt.title(f'f(x={int(x, 2)})={f(int(x, 2))}, T0={T0}')
plt.xlabel('iteration')
plt.ylabel('f(x)')
plt.plot(fs)

T0 = 100
cooling_coef = 0.9
x, fs = simulated_annealing(T0, cooling_coef, x)
plt.figure()
plt.title(f'f(x={int(x, 2)})={f(int(x, 2))}, T0={T0}')
plt.xlabel('iteration')
plt.ylabel('f(x)')
plt.plot(fs)

plt.show()

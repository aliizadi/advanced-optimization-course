import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


def f(W, c, sai: np.array):
    return 0.5 * np.linalg.norm(W) ** 2 + c * np.sum(sai)


def B(X, y, W, b, c, sai: np.array):
    out = 0
    for i in range(len(X)):
        out += np.log(y[i] * (np.dot(W, X[i]) + b) - 1 + sai[i]) + np.log(sai[i])

    return out


def objective(X, y, W, b, c, sai: np.array, epsilon):
    return f(W, c, sai) + epsilon * B(X, y, W, b, c, sai)


def grad_W(X, y, W, b, c, sai: np.array, epsilon):
    temp = np.zeros(len(W))
    for i in range(len(X)):
        temp += (y[i] * X[i]) / (y[i] * (np.dot(W, X[i]) + b) - 1 + sai[i])

    grad = W - epsilon * temp
    return grad


def grad_b(X, y, W, b, c, sai: np.array, epsilon):
    temp = 0
    for i in range(len(X)):
        temp += 1 / (sai[i] + 0.000001)

    grad = -epsilon * temp
    return grad


def grad_sai(X, y, W, b, c, sai: np.array, epsilon):
    grad = np.zeros(len(sai))
    grad += c - 2 * epsilon
    return grad


def interior_point_barrier(X_train, y_train):
    W = np.zeros(len(X_train[0]))
    b = 0
    sai = np.zeros(len(X_train))
    C = 1
    epsilon = 200
    alpha = 0.1

    fs = []

    for iteration in range(200):
        # if iteration % 10 == 0:
        #     print("Iteration: ", iteration)

        # for i in range(10000):
        w_grad = grad_W(X_train, y_train, W, b, C, sai, epsilon)
        b_grad = grad_b(X_train, y_train, W, b, C, sai, epsilon)
        sai_grad = grad_sai(X_train, y_train, W, b, C, sai, epsilon)

        W -= alpha * w_grad
        b -= alpha * b_grad
        sai -= alpha * sai_grad

        epsilon = epsilon * 0.1

        fs.append(f(W, C, sai))

    return W, b, sai, fs


def pred(x, w, b_e):
    out = w @ x + b_e
    if out >= 0:
        return 1
    else:
        return 0


def run():
    X, y = load_digits(return_X_y=True)

    X = X[y < 2]
    y = y[y < 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    W, b, sai, fs = interior_point_barrier(X_train, y_train)

    print(fs[-1])

    ccrs = []
    for x, y in zip(X_test, y_test):
        ccrs.append(abs(y - pred(x, W, b)))
    print(sum(ccrs) / (2 * len(X_test)))

    plt.plot(fs)
    plt.show()


run()

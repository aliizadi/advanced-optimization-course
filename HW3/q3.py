from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)

X = X[y < 2]
y = y[y < 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def auto_grad_f(x, w, b_e, b_d):
    w = torch.tensor(w, dtype=torch.float32, requires_grad=True)
    b_e = torch.tensor(b_e, dtype=torch.float32, requires_grad=True)
    b_d = torch.tensor(b_d, dtype=torch.float32, requires_grad=True)
    x = torch.tensor(x, dtype=torch.float32, requires_grad=False)

    w.grad = None
    b_e.grad = None
    b_d.grad = None

    x = x.reshape(x.shape[0], -1)
    error = torch.norm(x - w.T * (1 / (1 + torch.exp(-w @ x - b_e))) + b_d)
    error.backward(torch.ones(error.shape))
    return error.detach().numpy(), w.grad.detach().numpy(), b_e.grad.detach().numpy(), b_d.grad.detach().numpy()


def f(x, w, b_e, b_d):
    return np.linalg.norm(x - w.T * (1 / (1 + np.exp(-w @ x - b_e))) + b_d)


# increasing coefficient
v = 1.2

w = np.zeros((1, X_train.shape[1]))
b_e = np.zeros((1, 1))
b_d = np.zeros((X_train.shape[1], 1))

alpha = 0.001


def f_next(w, b_e, b_d, H_w, H_b_e, H_b_d, g_w, g_b_e, g_b_d, l):
    temp_H_w = H_w + l * np.diag(np.diag(H_w))
    temp_H_b_e = H_b_e + l * np.diag(np.diag(H_b_e))
    temp_H_b_d = H_b_d + l * np.diag(np.diag(H_b_d))
    # direction p
    p_w = - (np.linalg.pinv(temp_H_w) @ g_w.T).T
    p_b_e = - np.linalg.pinv(temp_H_b_e) @ g_b_e
    p_b_d = - np.linalg.pinv(temp_H_b_d) @ g_b_d
    errors = []
    for x in X_train:
        errors.append(f(x, w + alpha * p_w, b_e + alpha * p_b_e, b_d + alpha * p_b_d))
    return np.mean(errors)


def compute_f(w, b_e, b_d):
    errors = []
    for x in X_train:
        errors.append(f(x, w, b_e, b_d))
    return np.mean(errors)


l = 0.01

fs = []
ls = []

print(X_train.shape)

for iteration in range(300):
    if iteration % 100 == 0:
        print("Iteration: ", iteration)
    ws = np.zeros((1, X_train.shape[1]))
    b_es = np.zeros((1, 1))
    b_ds = np.zeros((X_train.shape[1], 1))

    H_ws = np.zeros((X_train.shape[1], X_train.shape[1]))
    H_b_es = np.zeros((1, 1))
    H_b_ds = np.zeros((X_train.shape[1], X_train.shape[1]))

    # print('step 1')
    for j, x_train in enumerate(X_train):
        error, w_grad, b_e_grad, b_d_grad = auto_grad_f(x_train, w, b_e, b_d)
        ws += error * w_grad
        b_es += error * b_e_grad
        b_ds += error * b_d_grad

        H_ws += w_grad.T @ w_grad
        H_b_es += b_e_grad.T @ b_e_grad
        H_b_ds += b_d_grad @ b_d_grad.T

    # gradient
    g_w = ws
    g_b_e = b_es
    g_b_d = b_ds

    # hessian
    H_w = H_ws
    H_b_e = H_b_es
    H_b_d = H_b_ds

    # adding lambda
    f_last = compute_f(w, b_e, b_d)
    l1 = deepcopy(l)
    l2 = l / v

    f_l1 = f_next(w, b_e, b_d, H_w, H_b_e, H_b_d, g_w, g_b_e, g_b_d, l1)
    f_l2 = f_next(w, b_e, b_d, H_w, H_b_e, H_b_d, g_w, g_b_e, g_b_d, l2)

    decreased_f_l1 = f_last - f_l1
    decreased_f_l2 = f_last - f_l2

    # levenberg-marquardt suggestion:

    if decreased_f_l1 < 0 and decreased_f_l2 < 0:
        l = l * v

    else:
        if decreased_f_l1 >= decreased_f_l2:
            l = l1

        else:
            l = l2

    # update
    H_w = H_w + l * np.diag(np.diag(H_w))
    H_b_e = H_b_e + l * np.diag(np.diag(H_b_e))
    H_b_d = H_b_d + l * np.diag(np.diag(H_b_d))

    p_w = - (np.linalg.pinv(H_w) @ g_w.T).T
    p_b_e = - np.linalg.pinv(H_b_e) @ g_b_e
    p_b_d = - np.linalg.pinv(H_b_d) @ g_b_d

    w += alpha * p_w
    b_e += alpha * p_b_e
    b_d += alpha * p_b_d
    # print('step 5')
    fs.append(compute_f(w, b_e, b_d))
    ls.append(l)


def pred(x, w, b_e):
    out = w @ x + b_e
    if out >= 0:
        return 1
    else:
        return 0


ccrs = []
for x, y in zip(X_test, y_test):
    ccrs.append(abs(y - pred(x, w, b_e)))

print(sum(ccrs) / (2 * len(X_test)))

plt.plot(fs)
plt.figure()
plt.plot(ls)
plt.show()

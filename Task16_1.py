import numpy as np
import sympy as sp
from functools import partial
import matplotlib.pyplot as plt
# from modules import *


def f(x):
    return x - 0.6


def delta(i, j):
    if i == j:
        return 1
    return 0


def u(x, c, alpha):
    s = 0
    for i in range(n):
        s += c[i] * alpha[i](x)
    return f(x) + s


def resolvent(x, y, D):
    res = 0
    for i in range(n):
        for j in range(n):
            res += abs(D[i, j]*alpha_sp[i](x)*beta_sp[j](y))
    return res


def apost_est(B, eta, u):
    coef = ((1 + B) * eta) / (1 - (1 + B) * eta)
    norm = 0
    for z in np.arange(start, end, 1e-4):
        if u(z) > norm:
            norm = u(z)

    return coef * norm


n = 4
start = 0
end = 1
nodes = np.array([start, (start + end) / 2, end])
alpha = [lambda x: 1, lambda x: np.square(x) / 2, lambda x: np.power(x, 4) / 4, lambda x: np.power(x, 6) / 16]
beta = [lambda y: 1, lambda y: np.square(y), lambda y: np.power(y, 4) / 6, lambda y: np.power(y, 6) / 45]

alpha_sp = [lambda x: 1, lambda x: (x ** 2) / 2, lambda x: (x ** 4) / 4, lambda x: (x ** 6) / 16]
beta_sp = [lambda y: 1, lambda y: y ** 2, lambda y: (y ** 4) / 6, lambda y: (y ** 6) / 45]

x = sp.Symbol('x', real=True)
y = sp.Symbol('y', real=True)

def get_c_vector(n):
    gamma = np.zeros((n, n))
    b = np.zeros(n)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gamma[i, j] = sp.integrate(beta_sp[i](y) * alpha_sp[j](y), (y, start, end))
            A[i, j] = delta(i, j) - gamma[i, j]
        b[i] = sp.integrate(beta_sp[i](y) * f(y), (y, start, end))

    D = np.linalg.inv(A)
    c = np.dot(D, b)

    return D, c


x1 = np.arange(start, end + 0.01, 0.01)

D, c = get_c_vector(n)
u_4 = u(nodes, c, alpha)
u_4_ext = u(x1, c, alpha)

n = 3
_, c = get_c_vector(n)
u_3 = u(nodes, c, alpha)
u_3_ext = u(x1, c, alpha)

d_ = np.max(np.abs(u_3 - u_4))
print("Значения решения в узлах при использовании ядра ранга 3: ", u_3)
print("Значения решения в узлах при использовании ядра ранга 4: ", u_4)
print("Модуль разности решений: ", d_)

B_sp = sp.integrate(resolvent(x, y, D), (y, start, end))
print(B_sp)
# def B_func(x): return 0.00396913169703231*x**6 + 0.164591999351516*x**4 + 2.91930102076639*x**2 + 16.6461147917504
def B_func(x): return 0.16458798743128*x**4 + 2.91921843390119*x**2 + 16.6456044356148


B_max = 0
for z in np.arange(start, end, 1e-4):
    if B_func(z) > B_max:
        B_max = B_func(z)

eta = 1 / 5040
u_x = partial(u, c=c, alpha=alpha)
print("Апостериорная оценка", apost_est(B_max, eta, u_x))
plt.plot(x1, u_3_ext - u_4_ext)
plt.show()


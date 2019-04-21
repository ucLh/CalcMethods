import numpy as np
from functools import partial
from modules import *

start = 0
end = 1
n = 200


# h = end - start
# nodes = np.arange(start, end, )


def f(x):
    return x - 0.6


def delta(i, j):
    if i == j:
        return 1
    return 0


def H(x, y): return np.cosh(x * y)


def integral_mid_rect(f, a, b, n):
    h = (b - a) / n
    s = 0
    for i in range(n):
        s += f(a + h / 2 + i * h)
    return h * s


def make_D_matrix(H, a, b):
    h = (b - a) / n
    d = np.zeros((n, n))
    x = np.zeros(n)
    for i in range(n):
        x[i] = a + h / 2 + i * h
        H_onedim = partial(H, x=x[i])
        for j in range(n):
            x_j = a + h / 2 + j * h
            d[i, j] = delta(i, j) - h * H_onedim(y=x_j)

    return d, x


def u(x, n, H, z, a=start, b=end):
    H_onedim = partial(H, x=x)
    s = 0
    h = (b - a) / n
    for i in range(n):
        x_i = a + h / 2 + i * h
        s += h * H_onedim(y=x_i) * z[i]
    return s + f(x)

D, nodes = make_D_matrix(H, start, end)
g = f(nodes)
z = solve_gauss_enhanced(D, g)

nodes1 = np.array([start, (start + end) / 2, end])
u_n = u(nodes1, n, H, z)
print("Решение, полученное при {} разбиениях: ".format(n), u_n)
n *= 2

D, nodes = make_D_matrix(H, start, end)
g = f(nodes)
z = solve_gauss_enhanced(D, g)

u_2n = u(nodes1, n, H, z)
print("Решение, полученное при {} разбиениях: ".format(n), u_2n)
d_ = np.max(np.abs(u_n - u_2n))
print("Расхождение решений: ", d_)

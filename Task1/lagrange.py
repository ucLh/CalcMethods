from functools import partial
from matplotlib import pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as pl
from numpy.polynomial.polynomial import Polynomial

plt.rcParams["patch.force_edgecolor"] = True
N = 13
x = np.array([-(N % 3), -(N % 3) + 0.1, -(N % 3) + 0.3, -(N % 3) + 0.45, -(N % 3) + 0.5])


def f_gen(x, n):
    return x * np.exp(x * (n % 2 + 1)) + np.sin(x * (n % 7 + 1) / 2)


def make_l_k(k, nodes, w, deriv):
    poly_part = np.polydiv(w.coef, np.array([-nodes[k], 1]))[0]
    return Polynomial(poly_part) / deriv(nodes[k])


def lagrange(nodes):
    result = Polynomial([0])
    w = Polynomial(np.poly(nodes)[::-1])
    deriv = w.deriv(1)
    for i in range(len(nodes)):
        result = pl.polyadd(result, make_l_k(i, nodes, w, deriv) * f(nodes[i]))[0]
    return result


f = partial(f_gen, n=N)
p = lagrange(x)

x1 = np.linspace(-1, -0.5)
plt.plot(x1, f(x1), label='f(x)')
plt.plot(x1, p(x1), label='p_n(x)')
plt.plot(x, f(x), 'o')
plt.legend(loc='upper left')
plt.show()

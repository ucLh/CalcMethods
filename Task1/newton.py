from functools import partial
from matplotlib import pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as pl
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import minimize
from scipy.misc import factorial
import sympy as sp
from sympy import Symbol, diff
from sympy.utilities import lambdify

plt.rcParams["patch.force_edgecolor"] = True
N = 13
x = np.array([-(N % 3), -(N % 3) + 0.1, -(N % 3) + 0.3, -(N % 3) + 0.45, -(N % 3) + 0.5])


def f_gen(x, n):
    return x * np.exp(x * ((n % 2) + 1)) + np.sin(x * ((n % 7) + 1) / 2)


def f_gen_sympy(x, n):
    return x * sp.exp(x * ((n % 2) + 1)) + sp.sin(x * ((n % 7) + 1) / 2)


def newton_interpol_polynomial(nodes, f):

    def divided_difference(row, col):
        # row = j, col = i
        if nodes[row + col] != nodes[row]:
            return (dif_matrix[row + 1, col - 1] - dif_matrix[row, col - 1]) / (nodes[row + col] - nodes[row])

    n = len(nodes)
    dif_matrix = np.zeros((n, n))

    for i in range(n):
        dif_matrix[i, 0] = f(nodes[i])

    for i in range(1, n):
        for j in range(0, n - i):
            dif_matrix[j, i] = divided_difference(j, i)

    result = Polynomial(dif_matrix[0, 0])

    for i in range(n - 1):
        result = pl.polyadd(result, Polynomial(np.poly(nodes[0:i + 1])[::-1]) * dif_matrix[0, i + 1])[0]

    return result, dif_matrix


def get_coef(f, c, d):
    n = 4
    y = Symbol('y')
    deriv = diff(f(y), y, y, y, y, y)
    func = lambdify(y, -abs(deriv))
    max_deriv = -minimize(func, [np.mean([c, d])], bounds=[(c, d)]).fun[0]
    return max_deriv / factorial(n + 1)


def max_error(x1, nodes, coef):
    w = Polynomial(np.poly(nodes)[::-1])
    return coef * abs(w(x1))


def chebyshev_roots(n, c, d):
    roots = []
    for i in range(n + 1):
        roots.append(np.cos(((2 * i + 1) * np.pi) / (2 * n + 2)))
    return list(map(lambda x: x * (d - c) / 2 + (d + c) / 2, roots))


f = partial(f_gen, n=N)
f_sp = partial(f_gen_sympy, n=N)
p, m = newton_interpol_polynomial(x, f)

x1 = np.linspace(-1, -0.5)
plt.plot(x1, f(x1), label='f(x)')
plt.plot(x1, p(x1), label='p_n(x)')
plt.plot(x, f(x), 'o')
plt.legend(loc='upper left')
plt.show()

coef = get_coef(f_sp, -1, -0.5)

plt.plot(x1, abs(f(x1) - p(x1)), label='fact_error')
plt.plot(x1, max_error(x1, x, coef), label='theor_error')
plt.legend(loc='upper left')
plt.show()
#
# x_cheb = chebyshev_roots(4, -1, -0.5)
# p_cheb, _ = newton_interpol_polynomial(x_cheb, f)
#
# print(x_cheb)
# plt.plot(x1, abs(f(x1) - p_cheb(x1)), label='fact_error')
# plt.plot(x1, max_error(x1, x_cheb, coef), label='theor_error')
# plt.legend(loc='upper left')
# plt.show()

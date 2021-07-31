from functools import partial
from matplotlib import pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as pl
from numpy.polynomial.polynomial import Polynomial
from scipy.misc import factorial
import sympy as sy
from sympy import Symbol, diff
from sympy.utilities import lambdify

plt.rcParams["patch.force_edgecolor"] = True
N = 13
x = np.array([-(N % 3), -(N % 3) + 0.3, -(N % 3) + 0.5])
y = np.array([-(N % 3), -(N % 3) + 0.3, -(N % 3) + 0.3, -(N % 3) + 0.5, -(N % 3) + 0.5, -(N % 3) + 0.5])


def f_gen(x, n):
    return x * np.exp(x * ((n % 2) + 1)) + np.sin(x * ((n % 7) + 1) / 2)


def f_gen_sympy(x, n):
    return x * sy.exp(x * ((n % 2) + 1)) + sy.sin(x * ((n % 7) + 1) / 2)


def derivative(f, order):
    if order < 0:
        return ValueError
    if order == 0:
        return f

    x = Symbol('x')
    deriv = diff(f(x), x)
    for i in range(1, order):
        deriv = sy.diff(deriv, x)
    return lambdify(x, deriv)


def hermit_interpol_polynomial(nodes, f):

    def divided_difference(row, col):
        # row = j, col = i
        # Если знаменатель не ноль
        if nodes[row + col] != nodes[row]:
            return (dif_matrix[row + 1, col - 1] - dif_matrix[row, col - 1]) / (nodes[row + col] - nodes[row])
        # Если знаменатель ноль
        else:
            deriv = derivative(f, col)
            return deriv(nodes[row]) / factorial(col)

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


f = partial(f_gen, n=N)
f_sy = partial(f_gen_sympy, n=N)
p, m = hermit_interpol_polynomial(y, f_sy)
x1 = np.linspace(-1, -0.5)

# Считаем производные функций и многочлена
t = Symbol('t')
d1_f = lambdify(t, diff(f_sy(t), t))
d1_p = lambdify(t, diff(p(t), t))
d2_f = lambdify(t, diff(f_sy(t), t, t))
d2_p = lambdify(t, diff(p(t), t, t))

plt.plot(x1, abs(f(x1) - p(x1)))
plt.title('Functions error')
plt.show()

y1 = [abs(d1_f(z) - d1_p(z)) for z in x1]
print(abs(d1_f(x[1]) - d1_p(x[1])))
print(abs(d1_f(x[2]) - d1_p(x[2])))
plt.plot(x1, y1)
plt.title('First order derivatives error')
plt.show()

y2 = [abs(d2_f(z) - d2_p(z)) for z in x1]
print(abs(d2_f(x[2]) - d2_p(x[2])))
plt.plot(x1, y2)
plt.title('Second order derivatives error')
plt.show()


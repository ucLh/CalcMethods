import numpy as np
from numpy.polynomial.polynomial import Polynomial
from numpy.linalg.linalg import solve
import sympy as sy
from sympy import Symbol, integrate

a, b = 0, 1  # границы отрезка, в котором мы работаем
nodes = [1 / 4, 1 / 2, 3 / 4]  # Заданные узлы
INTEGRAL_VALUE = 0.5284080812266490  # Точное значение интеграла


def p(x):
    return np.power(x, -1 / 4)


def f(x):
    return sy.sin(x)


def make_a_k(nodes, p, a, b):
    w = Polynomial(np.poly(nodes)[::-1])
    result_coeffs = []
    x = Symbol("x")
    for k in range(len(nodes)):
        poly_part = np.polydiv(w.coef, np.array([-nodes[k], 1]))[0] / w.deriv()(nodes[k])

        # Вручную домножаем каждый член многоочлена на весовую функцию, чтобы потом взять интеграл
        under_integral_list = []
        for i, coef in enumerate(poly_part):
            under_integral_list.append(coef * x**i * p(x))
        under_integral = sum(under_integral_list)

        # Берём интеграл на промежутке [a,b]
        result_coeffs.append(integrate(under_integral, (x, a, b)))
    return result_coeffs


def max_ast_integral(nodes, f, p, a, b):
    """ Вычисляет интеграл произведения ф-ций f и p"""
    a_k = make_a_k(nodes, p, a, b)
    result = 0
    for k, x in enumerate(nodes):
        result += a_k[k] * f(nodes[k])
    return result


def make_mu_k(n, p, a, b):
    x = Symbol("x")
    return [integrate(p(x) * (x**i), (x, a, b)) for i in range(2*n)]


def make_nodes(n, p, a, b):
    mu_list = np.array(make_mu_k(n, p, a, b), dtype=np.float64)
    left_part = np.array([mu_list[i:n+i][::-1] for i in range(n)], dtype=np.float64)
    right_part = -1 * mu_list[n:]
    coef = solve(left_part, right_part)
    coef = np.append(coef[::-1], 1)

    w = Polynomial(coef)
    return w.roots()


print("Погрешность при использовании интерпол. квадратурной формулы с фиксированными узлами")
print(abs(max_ast_integral(nodes, f, p, a, b) - INTEGRAL_VALUE))

optim_nodes = make_nodes(3, p, a, b)
print("Погрешность при использовании квадратурной формулы с наивысшей АСТ")
print(abs(max_ast_integral(optim_nodes, f, p, a, b) - INTEGRAL_VALUE))

from functools import partial
import logging
import numpy as np
from numpy.polynomial import polynomial as pl
from numpy.polynomial.polynomial import Polynomial
import sympy as sp
from sympy import Symbol
from Task1.newton import newton_interpol_polynomial

N = 13
h = 0.01  # шаг между узлами
dots_list = [-(N % 3) + i for i in np.arange(0, 1 + h, h)]  # пространство точек
dots = np.array(dots_list)
m = len(dots) - 1  # индекс последней точки пространства
logging.basicConfig(filename=u'log.txt', level=logging.INFO)


def f_gen(x, n):
    return x * np.exp(x * ((n % 2) + 1)) + np.sin(x * ((n % 7) + 1) / 2)


def make_nodes(quantity, x_init, dots):
    h = dots[1] - dots[0]
    m = len(dots) - 1
    if x_init < dots[0] or x_init > dots[m]:
        raise ValueError()

    def get_from_start(quant):
        return dots[:quant]

    def get_from_end(quant):
        return dots[-quant:][::-1]

    # Вычисляем индекс ближайшего узла к исходной точке
    def get_neighbour_index(x_init):
        neighbour = min(dots, key=lambda x: abs(x - x_init))
        return dots.tolist().index(neighbour)

    def check_right_bound(neigh_ind, quant):
        return (neigh_ind + quant // 2) >= m

    def check_left_bound(neigh_ind, quant):
        return (neigh_ind - (quant - 1) // 2) <= 0

    n_ind = get_neighbour_index(x_init)
    if (dots[0] < x_init <= dots[0] + h / 2) or check_left_bound(n_ind, quantity):
        result = get_from_start(quantity)
        ind = 0
    elif dots[m] - h / 2 <= x_init < dots[m] or check_right_bound(n_ind, quantity):
        result = get_from_end(quantity)
        ind = quantity - 1
    else:
        result = dots[n_ind - (quantity - 1) // 2:n_ind + (quantity // 2) + 1]
        ind = result.tolist().index(dots[n_ind])

    # ind - это x_0 в случаях 1 и 3, x_m в случае 2
    return result, ind


def finite_difference(f, x, order, step):
    if order < 1:
        raise ValueError()
    if order == 1:
        return f(x + step) - f(x)
    elif order > 1:
        return finite_difference(f, x + step, order - 1, step) - finite_difference(f, x, order - 1, step)


def n_k(order, coef_func):
    if order < 0:
        raise ValueError()
    if order == 0:
        return Polynomial([0, 1])
    elif order > 0:
        return pl.polymul(n_k(order - 1, coef_func), Polynomial([coef_func(order), 1]))[0] / (order + 1)


def diff_matrix(f, nodes):
    n = len(nodes)
    dif_matrix = np.zeros((n, n))

    for i in range(n):
        dif_matrix[i, 0] = f(nodes[i])

    for i in range(1, n):
        for j in range(0, n - i):
            dif_matrix[j, i] = (dif_matrix[j + 1, i - 1] - dif_matrix[j, i - 1])

    return dif_matrix


def _process_diff_matrix(dm, flag, start_index):
    """
    Выюирает нужные конечные разности из матрицы конечных разностей
    """
    n = dm.shape[0]
    if flag == 0:
        result = dm[0, :]
    if flag == 1:
        result = list()
        for i in range(n):
            result.append(dm[n - i - 1, i])
    if flag == 2:
        result = list()
        result.append(dm[start_index, 0])
        for i in range(0, n - 1):
            result.append(dm[start_index - ((i + 1) // 2), i + 1])

    return result


def newton_equal_dist(f, nodes, start_ind):
    n = len(nodes) - 1
    h = nodes[1] - nodes[0]
    result_t = Polynomial(f(nodes[start_ind]))
    dm = diff_matrix(f, nodes)

    # Разбор случаев расположения исходной точки относительно узлов
    if start_ind == 0:
        flag = 0

        def coef_func(x):
            return -x
    elif start_ind == len(nodes) - 1:
        flag = 1

        def coef_func(x):
            return x
    else:
        flag = 2

        def coef_func(x):
            return ((-1) ** x) * ((x + 1) // 2)

    fin_dif_list = _process_diff_matrix(dm, flag, start_ind)

    for i in range(n):
        sum_part = n_k(i, coef_func) * fin_dif_list[i + 1]
        result_t = pl.polyadd(result_t, sum_part)[0]

    # Производим замену переменной
    x, t = Symbol('x'), Symbol('t')
    t = (x - nodes[start_ind]) / h
    # Переделываем полином из numpy в полином из sympy, подставляя x вместо t
    result_t = sp.Poly.from_list(result_t.coef[::-1], gens=t)

    # Раскрываем скобки
    result = sp.simplify(result_t)

    # Возвращаем полином из numpy
    return Polynomial(result.all_coeffs()[::-1])


f = partial(f_gen, n=N)
print("Введите точку х: ")
x_ = float(input())
print("Введите степень полинома: ")
order = int(input())

if not ((dots[0] <= x_ <= dots[m]) and (0 <= order <= m)):
    raise ValueError()

nodes, index = make_nodes(order + 1, x_, dots)
print(nodes, index)
p = newton_equal_dist(f, nodes, index)
print(p)
# p_old, _ = newton_interpol_polynomial(nodes, f)
# print(p_old)
print(abs(f(x_) - p(x_)))

logging.info(" x: " + str(x_) + " n: " + str(order) +
             "\nnodes: " + str(nodes) + "index: " + str(index) +
             "\npolynomial: " + str(p) +
             "\nerror: " + str(abs(f(x_) - p(x_))))

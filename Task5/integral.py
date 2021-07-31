from collections import namedtuple
from functools import partial
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
import sympy as sy
from sympy import Symbol, diff
from sympy.utilities import lambdify

plt.rcParams["patch.force_edgecolor"] = True
a, b = 0, 1
C = 5


def f_gen(x, c):
    return 1 / (np.square(x) + c)


f = partial(f_gen, c=C)


def integral_f():
    return np.arctan(1 / np.sqrt(5)) / np.sqrt(5)


def make_nodes(a, b, n):
    h = (b - a) / n
    return [a + k * h for k in range(n+1)]


def integral_num_f(n, config=None):
    # n = len(nodes) - 1
    result = 0.0
    if config is None:
        config = {'type': 'rect', 'subtype': 'right'}
    if config['type'] == 'rect':
        h = (b - a) / n
        if config['subtype'] == 'right':
            delta = a + h
        elif config['subtype'] == 'left':
            delta = a
        elif config['subtype'] == 'mid':
            delta = a + h / 2
        else:
            raise KeyError()

        for i in range(1, n + 1):
            result += h * (f(delta + (i - 1) * h))

    elif config['type'] == 'trap':
        nodes = make_nodes(a, b, n)
        result += f(nodes[0])
        for i in range(1, n):
            result += 2 * f(nodes[i])
        result += f(nodes[n])
        result *= (b - a) / (2 * n)

    elif config['type'] == 'simpson':
        # h = (b - a) / (2 * n)
        nodes = make_nodes(a, b, 2*n)
        for i in range(1, n):
            result += 4 * f(nodes[2*i-1]) + 2 * f(nodes[2*i])
        result += 4 * f(nodes[2*n-1]) + f(nodes[2*n]) + f(nodes[0])
        result *= (b - a) / (6 * n)

    else:
        raise KeyError()

    return result


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


def max_error(n, config):
    if config['type'] == 'rect':
        if config['subtype'] == 'right' or config['subtype'] == 'left':
            C = 1 / 2
            d = 0
        elif config['subtype'] == 'mid':
            C = 1 / 24
            d = 1
        else:
            raise KeyError()
    elif config['type'] == 'trap':
        C = 1 / 12
        d = 1
    elif config['type'] == 'simpson':
        C = 1 / 2880
        d = 3
    else:
        raise KeyError()

    deriv = derivative(f, d+1)
    coef = C * (b - a) * np.power((b - a) / n, d + 1)
    func = lambda x: -abs(deriv(x))
    max_deriv = -minimize(func, np.mean([a, b]), bounds=[(a, b)]).fun[0]
    return coef * max_deriv


print('Введите n (кол-во отрезков в разбиении):')
N = int(input())
print('Точное значение интеграла:', integral_f())

print('Фактическая и теоретическая ошибки:')

configs = {
    'types': ('rect', 'trap', 'simpson'),
    'subtypes': {'rect': ('right', 'left', 'mid')}
}

for type in configs['types']:
    cfg = dict()
    cfg['type'] = type
    if type in configs['subtypes']:
        for subt in configs['subtypes'][type]:
            cfg['subtype'] = subt
            print(type, subt)
            print(abs(integral_num_f(N, cfg) - integral_f()), max_error(N, cfg))
    else:
        cfg['subtype'] = None
        print(type)
        print(abs(integral_num_f(N, cfg) - integral_f()), max_error(N, cfg))

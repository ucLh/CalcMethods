from functools import partial
import numpy as np

from num_integral import integral_num_f

a, b = 0, 1
C = 5
# roots of legendre polynomial of 5-th order
l_r = (-0.906179845938664, -0.538469310105683, 0, 0.538469310105683, 0.906179845938664)
# gaussian coefficients, A_k
g_c = (0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189)


def f_gen(x, c):
    return 1 / (np.square(x) + c)


f = partial(f_gen, c=C)


def integral_f():
    return np.arctan(1 / np.sqrt(C)) / np.sqrt(C)


def gauss_integral_formula(l_r, g_c):
    # change the variable
    result = 0
    nodes = [t * (b - a) / 2 + (b + a) / 2 for t in l_r]
    for i in range(len(nodes)):
        result += g_c[i] * f(l_r[i])
    return result * (b - a) / 2


print("Погрешность при использовании формулы Гаусса: ", abs(gauss_integral_formula(l_r, g_c) - integral_f()))

cfg_s = {'type': 'simpson'}
num_nodes = 2  # so we will make 5 nodes
print("Погрешность при использовании формулы Симпсона: ", abs(integral_num_f(num_nodes, cfg_s) - integral_f()))

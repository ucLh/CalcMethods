from enum import Enum
from functools import partial
from matplotlib import pyplot as plt
import numpy as np
from num_integral import derivative
from scipy.optimize import minimize
import sympy as sy

plt.rcParams["patch.force_edgecolor"] = True


class Types(Enum):
    newt = "newton"
    newt_v2 = "newton_v2"
    newt_v3 = "newton_v3"


def f(x):
    return np.log(x) + 2 * x - 3


def f_sy(x):
    return sy.log(x) + 2 * x - 3


def first_approx(f, a, b):
    df2 = derivative(f_sy, 2)
    if f(b) * df2(b) > 0:
        return b
    elif f(a) * df2(a) > 0:
        return a


def get_next_approx_newton(f, x_k, df):
    return x_k - (f(x_k) / df(x_k))


def get_next_approx_newton_v2(f, x_k, df0):
    """Метод использующий производную только в точке первого приближения"""
    return x_k - (f(x_k) / df0)


def get_next_approx_newton_v3(f, prev_x, cur_x):
    """Метод использующий численное дифференцирование"""
    return cur_x - (f(cur_x) * (cur_x - prev_x)) / (f(cur_x) - f(prev_x))


def newton_method(f, a, b, eps=1e-5, cfg=None):
    prev_x = first_approx(f, a, b)
    df = derivative(f, 1)

    if cfg is None:
        cfg = {"type": Types.newt.value}

    get_next_func, cur_x, iter_count = None, None, None
    # Задаем функцию взятия следующего элемента для разных способов
    if cfg["type"] == Types.newt.value:
        get_next_func = partial(get_next_approx_newton, df=df)
        cur_x = get_next_func(f, prev_x)
    elif cfg["type"] == Types.newt_v2.value:
        get_next_func = partial(get_next_approx_newton_v2, df0=df(prev_x))
        cur_x = get_next_func(f, prev_x)
    elif cfg["type"] == Types.newt_v3.value:
        if a <= (prev_x + 2 * eps) <= b:
            cur_x = prev_x + 2 * eps
        elif a <= (prev_x - 2 * eps) <= b:
            cur_x = prev_x - 2 * eps

    # Итерационный процесс
    iter_count = 1
    while np.abs(prev_x - cur_x) >= eps:
        if cfg["type"] == Types.newt_v2.value or cfg["type"] == Types.newt.value:
            prev_x = cur_x
            cur_x = get_next_func(f, prev_x)
        elif cfg["type"] == Types.newt_v3.value:
            next_x = get_next_approx_newton_v3(f, prev_x, cur_x)
            prev_x = cur_x
            cur_x = next_x
        iter_count += 1
        assert a <= cur_x <= b
    return cur_x, iter_count


def simple_iteration_method(f, a, b, eps=1e-5):
    df = derivative(f, 1)

    def find_tau():
        max_deriv = -minimize(lambda x: -df(x), np.mean([a, b]), bounds=[(a, b)]).fun[0]
        return 1 / max_deriv

    tau = find_tau()

    def phi(x):
        return x - tau * f(x)

    # q = 1 - minimize(df, np.mean([a, b]), bounds=[(a, b)]).fun[0] * tau
    dphi = derivative(phi, 1)
    q = -minimize(lambda x: -abs(dphi(x)), np.mean([a, b]), bounds=[(a, b)]).fun[0]
    # Инициализируем x0
    prev_x = np.mean([a, b])
    cur_x = phi(prev_x)
    iter_count = 1
    while np.abs(cur_x - prev_x) >= (eps * (1 - q) / q):
        prev_x = cur_x
        cur_x = phi(prev_x)
        iter_count += 1
        assert a <= cur_x <= b

    return cur_x, iter_count


# Локализуем корни
x1 = np.linspace(1e-3, 5, 200)
plt.plot(x1, f(x1))
x2 = np.array([1e-3, 1, 2, 3, 4, 5])
plt.plot(x2, f(x2), 'o')
plt.plot(x1, [0] * 200)
# plt.show()
# Можем взять промежуток локализации [1, 2]
print("f(1): {}, f(2): {}".format(f(1), f(2)))
a, b = 1, 2
# Указываем эпсилон здесь
eps = 1e-10

types = ["newton", "newton_v2", "newton_v3"]
for type in types:
    cfg = {"type": type}
    root, count = newton_method(f_sy, a, b, eps=eps, cfg=cfg)
    print("type: {}, f(x): {}, iterations: {}".format(type, f(float(root)), count))
root, count = simple_iteration_method(f_sy, a, b, eps=eps)
print("type: simple_iter, f(x): {}, iterations: {}".format(f(float(root)), count))

# plt.plot(root, f_sy(root), 'o')
# plt.show()



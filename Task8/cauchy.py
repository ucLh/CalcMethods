from enum import Enum
import numpy as np
from num_integral import make_nodes

a, b = 1, 2
y0 = 0
N = 1000


class Types(Enum):
    euler = "euler"
    euler_mid = "euler_mid"
    euler_trapeze = "euler_trapeze"
    runge_kutta = "runge_kutta"


def f(x, y):
    """
    differential equation
    Cauchy task: y(1) = 0
    x in [1, 2]
    """
    return 2 * y / x + x


def analytical_solution(x):
    return np.log(x) * np.square(x)


def adams_method(f, a, b, nodes):
    n = len(nodes) - 1
    h = (b - a) / n

    def q(k):
        return f(nodes[k], y[k]) * h

    y_ = solve_cauchy_task(f, a, b, nodes, cfg={'type': Types.runge_kutta.value}).tolist()
    y = y_[:5]
    for i in range(4, n):
        y.append(y[i] + (1901 * q(i) - 2774 * q(i - 1) + 2616 * q(i - 2) - 1274 * q(i - 3) + 251 * q(i - 4)) / 720)

    return np.array(y)


def solve_cauchy_task(f, a, b, nodes, cfg=None):
    n = len(nodes) - 1
    h = (b - a) / n
    y = [y0]

    if cfg is None:
        cfg = {'type': 'euler'}

    for i in range(0, n):
        # Uses left-rectangular formula for integral approx
        if cfg['type'] == Types.euler.value:
            y.append(y[i] + h * f(nodes[i], y[i]))
        # Uses mid-rectangular formula for integral approx
        elif cfg['type'] == Types.euler_mid.value:
            y.append(y[i] + h * f(nodes[i] + h / 2, y[i] + h * (f(nodes[i], y[i])) / 2))
        # Uses trapeze formula for integral approx
        elif cfg['type'] == Types.euler_trapeze.value:
            y_hat = y[i] + h * f(nodes[i], y[i])
            y.append(y[i] + h * (f(nodes[i], y[i]) + f(nodes[i + 1], y_hat)) / 2)
        # Uses Simpson formula for integral approx
        elif cfg['type'] == Types.runge_kutta.value:
            k1 = h * f(nodes[i], y[i])
            k2 = h * f(nodes[i] + h / 2, y[i] + k1 / 2)
            k3 = h * f(nodes[i] + h / 2, y[i] + k2 / 2)
            k4 = h * f(nodes[i] + h, y[i] + k3)
            y.append(y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)

    return np.array(y)


nodes = make_nodes(a, b, N)
precise = analytical_solution(nodes)
types = ("euler", "euler_mid", "euler_trapeze", "runge_kutta")

for t in types:
    cfg = {"type": t}
    print(t, abs(precise - solve_cauchy_task(f, a, b, nodes, cfg=cfg)))

print("adams", abs(precise - adams_method(f, a, b, nodes)))





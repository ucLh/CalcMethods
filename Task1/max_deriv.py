import numpy as np
from scipy.optimize import minimize
from scipy.misc import factorial
from sympy import Symbol, diff
from sympy.utilities import lambdify


def get_coef(f, c, d):
    n = 4
    y = Symbol('y')
    deriv = diff(f(y), y, y, y, y, y)
    func = lambdify(y, -abs(deriv))
    max_deriv = -minimize(func, [np.mean([c, d])], bounds=[(c, d)]).fun[0]
    return max_deriv / factorial(n + 1)


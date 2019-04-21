from functools import partial
import numpy as np
from numpy.linalg import det
from numpy.linalg import norm, solve, inv


A = np.array([[-401.52, 200.16], [1200.96, -601.68]])
b = np.array([200, -600])
b_ = np.array([-1, -1])
norm_inf = partial(norm, ord=np.inf)


def addition(A, i, j):
    """
    :param A: Матрица
    :param i: Номер строки
    :param j: Номер столбца
    :return: Алгебраическое дополнение элемента на позиции (i, j)
    """
    A_shortened = np.delete(A, i, 0)
    A_shortened = np.delete(A_shortened, j, 1)
    return (-1) ** (i + j) * det(A_shortened)


def invert(A):
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    A_inv = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A_inv[i, j] = addition(A, i, j)
    return A_inv.transpose() / det(A)

def condition_number(A):
    return norm_inf(A) * norm_inf(invert(A))


def get_delta(x, d_x):
    return norm_inf(d_x)/norm_inf(x)


x = solve(A, b)
x_ = solve(A, b + b_)
delta_x = norm_inf(x_ - x)/norm_inf(x)
delta_b = get_delta(b, b_)

print("Число обусловленности матрицы: ", condition_number(A))
print("Фактическая относительная погрешность: ", delta_x)
print("Оценка относительной погрешности числом обучловленности: ", condition_number(A) * delta_b)

# B = np.array([[0, 1, -3], [3, 1, 2], [-2, -1, 4]])
# print(np.allclose(invert(A), inv(A)))


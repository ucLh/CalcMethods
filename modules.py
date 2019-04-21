import copy
from enum import Enum
from functools import partial
import numpy as np
from numpy.linalg import det
from numpy.linalg import norm

norm_inf = partial(norm, ord=np.inf)


class GaussOptions(Enum):
    column = 0
    row = 1
    matrix = 2


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


def solve_gauss(A, b, epsilon=1e-8):
    assert A.shape[0] == A.shape[1] == len(b)
    n = A.shape[0]
    A = np.append(A, b[:, np.newaxis], axis=1)

    # forward pass
    for k in range(n):
        tmp = A[k, k]
        if abs(tmp) < epsilon:
            raise ZeroDivisionError()
        for j in range(k, n+1):
            A[k, j] /= tmp
        for i in range(k + 1, n):
            tmp = A[i, k]
            for j in range(k, n+1):
                A[i, j] = A[i, j] - A[k, j] * tmp

    # backward pass
    x = np.zeros(n)
    for i in range(n-1, 0-1, -1):
        x[i] = A[i, n]
        for j in range(i+1, n):
            x[i] -= A[i, j] * x[j]

    return x


def _choose_main_elements(A, cfg):
    assert A.shape[0] == A.shape[1]
    # in a column
    n = A.shape[0]
    ord_x = np.arange(n)
    ord_b = np.arange(n)
    if cfg == GaussOptions.column.value:
        for i in range(n-1):
            ind = np.argmax(np.abs(A[i:, i])) + i
            tmp = copy.deepcopy(A[i, :])
            A[i, :], A[ind, :] = A[ind, :], tmp
            ord_b[i], ord_b[ind] = ord_b[ind], ord_b[i]
    # in a row
    elif cfg == GaussOptions.row.value:
        for i in range(n-1):
            ind = np.argmax(np.abs(A[i, i:])) + i
            tmp = copy.deepcopy(A[:, i])
            A[:, i], A[:, ind] = A[:, ind], tmp
            ord_x[i], ord_x[ind] = ord_x[ind], ord_x[i]
    # in the whole matrix
    elif cfg == GaussOptions.matrix.value:
        for k in range(n-1):
            i, j = np.unravel_index(np.abs(A[k:, k:]).argmax(), A[k:, k:].shape)  # i - row, j - column
            i += k
            j += k

            tmp = copy.deepcopy(A[k, :])
            A[k, :], A[i, :] = A[i, :], tmp
            ord_b[k], ord_b[i] = ord_b[i], ord_b[k]

            tmp = copy.deepcopy(A[:, k])
            A[:, k], A[:, j] = A[:, j], tmp
            ord_x[k], ord_x[j] = ord_x[j], ord_x[k]

    return A, ord_x, ord_b


def solve_gauss_enhanced(A, b, epsilon=1e-8, cfg=GaussOptions.matrix.value):
    assert A.shape[0] == A.shape[1] == len(b)

    def restore_order(a, ord_a):
        tmp = copy.deepcopy(a)
        for i in range(len(ord_a)):
            a[ord_a[i]] = tmp[i]
        return a

    A_, ord_x, ord_b = _choose_main_elements(copy.deepcopy(A), cfg)
    x = solve_gauss(A_, b[ord_b], epsilon)
    return restore_order(x, ord_x)

import numpy as np
from numpy.linalg import eig, norm
from modules import *

# A = np.array([[12.951443, 1.554567, -3.998582], [1.554567, 9.835076, 0.930339], [-3.998582, 0.930339, 7.80380]])
# A = np.array([[-0.81417, -0.01937, 0.41372], [-0.01937, 0.54414, 0.00590], [0.41372, 0.00590, -0.81445]])

A = np.array([[-1.00449, -0.38726, 0.59047],
[-0.38726, 0.73999, 0.12519],
[0.59047, 0.12519, -1.08660]])

def Jacobi(A, eps):
    def rotate(A, V, ik, jk):
        n = A.shape[0]
        aDiv = A[ik][ik] - A[jk][jk]
        phi = 0.5 * np.arctan(2 * A[ik][jk] / aDiv)
        c = np.cos(phi)
        s = np.sin(phi)
        for i in range(n):
            if (i != ik) and (i != jk):
                A[ik][i] = c * A[i][ik] + s * A[i][jk]
                A[jk][i] = (-1) * s * A[i][ik] + c * A[i][jk]
                A[i][ik] = A[ik][i]
                A[i][jk] = A[jk][i]
        temp1 = (c ** 2) * A[ik][ik] + 2 * c * s * A[ik][jk] + (s ** 2) * A[jk][jk]
        temp2 = (s ** 2) * A[ik][ik] - 2 * c * s * A[ik][jk] + (c ** 2) * A[jk][jk]
        A[ik][ik] = temp1
        A[jk][jk] = temp2
        A[ik][jk] = 0.0
        A[jk][ik] = 0.0
        for i in range(n):
            temp1 = c * V[i][ik] + s * V[i][jk]
            temp2 = (-1) * s * V[i][ik] + c * V[i][jk]
            V[i][ik] = temp1
            V[i][jk] = temp2

    n = A.shape[0]
    V = np.identity(n) * 1.0
    # B = copy.deepcopy(A)
    def over_diagonal_argmax(A):
        n = A.shape[0]
        D = np.zeros_like(A)
        for i in range(n - 1):
            for j in range(i + 1, n):
                D[i][j] = A[i][j]
        ik, jk = np.unravel_index(np.abs(D).argmax(), D.shape)  # i - row, j - column
        assert ik < jk
        return A[ik][jk], ik, jk

    current, ik, jk = over_diagonal_argmax(A)
    while np.abs(current) >= eps:
        rotate(A, V, ik, jk)
        current, ik, jk = over_diagonal_argmax(A)

    return np.diagonal(A), V


def power_method(A, eps, Yk=None):
    if Yk is None:
        Yk = np.array([-0.01, -0.01, -0.01])
    res = 1
    k = 0
    p = np.argmax(np.abs(Yk))
    while res >= eps:
        k += 1
        Yk = Yk / Yk[p]
        Yk_next = np.dot(A, Yk)
        l1 = Yk_next[p] / Yk[p]
        res = norm(np.dot(A, Yk_next) - l1 * Yk_next, np.inf)
        Yk = Yk_next
        if k >= 500:
            print("Does not converge")
            return 0, 0, k
    return l1, Yk / norm(Yk), k


def scal_prod(A, eps, Yk=None):
    if Yk is None:
        Yk = np.array([-0.01, -0.01, -0.01])
    res = 1
    k = 0
    p = np.argmax(np.abs(Yk))
    while res >= eps:
        k += 1
        Yk = Yk / Yk[p]
        Yk_next = np.dot(A, Yk)
        l1 = np.dot(Yk_next, Yk) / np.dot(Yk, Yk)
        res = norm(np.dot(A, Yk_next) - l1 * Yk_next, np.inf)
        Yk = Yk_next
        if k >= 500:
            print("Does not converge")
            return 0, 0, k
    return l1, Yk / norm(Yk), k


def spec_bound(A, eps):
    l1, l1_vec, _ = power_method(A, eps)
    B = A - l1 * np.identity(A.shape[0])
    lB, lB_vec, _ = power_method(B, eps)
    res_vec = (l1_vec + lB_vec) / norm(l1_vec + lB_vec)
    return lB + l1, res_vec


def wielandt_refinement(A, eps):
    lk = -1
    k = 0
    res = 1
    # start backward iterations
    while res >= eps:
        k += 1
        W = A - lk * np.identity(A.shape[0])
        l, v, i = scal_prod(invert(W), eps)
        lk = 1 / l + lk
        res = norm(np.dot(A, v) - lk * v)
        if k >= 500:
            print("Does not converge")
            return 0, 0, k
    return lk, v, k


# 1)
# print("Real eigenvalues:\n", eig(A))
C = copy.deepcopy(A)
l, x = Jacobi(C, 1e-6)
print("Jacobi method:\n", l, x)
print("Norms of vectors: ")
for i in range(len(x)):
    print(norm(x[:, i]))
print("Residual: ", np.dot(A, x)-l*x)
print()

# 2)
l, x, k = power_method(A, 1e-3)
print("Power method:\n", l, x)
print("Number of iterations: ", k)
print("Residual: ", np.dot(A, x)-l*x)
print("Norm of vector: ", norm(x))
print()

# 3)
l, x, k = scal_prod(A, 1e-3)
print("Scalar product method:\n", l, x)
print("Number of iterations: ", k)
print("Residual: ", np.dot(A, x) - l*x)
print("Norm of vector: ", norm(x))
print()

# 4)
l, x = spec_bound(A, 1e-3)
print("Spectre bound:\n", l, x)
print("Residual: ", np.dot(A, x) - l*x)
print("Norm of vector: ", norm(x))
print()

# 5)
l, x, k = wielandt_refinement(A, 1e-3)
print("Wielandt method:\n", l, x)
print("Residual: ", np.dot(A, x) - l*x)
print("Norm of vector: ", norm(x))
print(k)

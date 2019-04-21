from numpy.linalg import eig
from modules import *


A = np.array([[12.951443, 1.554567, -3.998582], [1.554567, 9.835076, 0.930339], [-3.998582, 0.930339, 7.80380]])
b = np.array([4.03171, 11.5427, 6.73485])


def transfer_system(A, b):
    assert A.shape[0] == A.shape[1] == len(b)
    H = np.zeros_like(A)
    g = np.zeros_like(b)
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                H[i, j] = 0
            else:
                H[i, j] = -A[i, j] / A[i, i]
        g[i] = b[i] / A[i, i]

    return H, g


def simple_iteration(H, g, k, x0=None):
    assert H.shape[0] == H.shape[1] == len(g)
    if x0 is None:
        x0 = np.zeros_like(g)
    x_cur = x0
    for i in range(k):
        x_cur = np.dot(H, x_cur) + g
    return x_cur


def apost_est(x_k, x_j, H):
    return norm_inf(H) * norm_inf(x_k - x_j) / (1 - norm_inf(H))


def seidel(H, g, k, x0=None):
    assert H.shape[0] == H.shape[1] == len(g)
    if x0 is None:
        x0 = np.zeros_like(g)
    n = H.shape[0]
    Hl, Hr = np.zeros_like(H), np.zeros_like(H)
    for i in range(n):
        Hr[i, i:] = H[i, i:]
        Hl[i, :i] = H[i, :i]
    E = np.eye(n)
    He = invert(E - Hl)

    x_cur = x0
    for i in range(k):
        x_cur = np.dot(He, np.dot(Hr, x_cur)) + np.dot(He, g)

    return x_cur, np.dot(He, Hr)


def spectre_rad(A):
    eig_numbers = eig(A)[0]
    return np.max(np.abs(eig_numbers))


def lusternik_correction(x_k, x_j, H):
    return x_j + (x_k - x_j) / (1 - spectre_rad(H))


def relaxation(H, g, k, x0=None):
    assert H.shape[0] == H.shape[1] == len(g)
    if x0 is None:
        x0 = np.zeros_like(g)
    n = H.shape[0]
    sr = spectre_rad(H)
    q = 2.0 / (1 + np.sqrt(1 - sr**2))
    x_cur = x0
    for l in range(k):
        x_next = np.zeros_like(x_cur)
        for i in range(n):
            s1 = 0
            for j in range(i):
                s1 += (H[i][j] * x_next[j])
            s2 = 0
            for j in range(i, n):
                s2 += (H[i][j] * x_cur[j])
            x_next[i] = x_cur[i] + q * (g[i] - x_cur[i] + s1 + s2)
        x_cur = x_next

    return x_cur


"""Variant 6"""
# 1)
x_star = solve_gauss_enhanced(A, b)
print("Точное решение: ", x_star)

# 2)
H, g = transfer_system(A, b)
print("Норма матрицы H: ", norm_inf(H))

# 3)
x_10 = simple_iteration(H, g, 10)
# print("Решение методом простой итерации", x_10)

# 4)
x_9 = simple_iteration(H, g, 9)
print("Апостериорная оценка для метода простой итерации: ", apost_est(x_10, x_9, H))
print("Фактическая погрешность для метода простой итерации: ", norm_inf(x_10 - x_star))

# 5)
x_10_seid, H_seid = seidel(H, g, 10)
# print(x_10_seid)
print("Фактическая погрешность для метода Зейделя: ", norm_inf(x_10_seid - x_star))

# 6)
print("Спектральный радиус матрицы перехода: ", spectre_rad(H_seid))
x_9_seid = seidel(H, g, 9)[0]
x_l = lusternik_correction(x_10, x_9, H)
print("Уточнение Люстерника для приближения по методу простой итерации: ", norm_inf(x_l - x_star))
x_l = lusternik_correction(x_10_seid, x_9_seid, H)
print("Уточнение Люстерника для приближения по методу Зейделя: ", norm_inf(x_l - x_star))
# 7)
x_10_rel = relaxation(H, g, 10)
# print(x_10_rel)
print("Фактическая погрешность для метода верхней релаксации: ", norm_inf(x_10_rel - x_star))
print(H)

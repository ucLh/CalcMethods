import numpy as np

n = 10
a, b = -1, 1
h = (b - a) / n
nodes = np.arange(a, b + h, h)

# u(-1) = u(1) = 0


def p(x):
    return 1 / (x - 3)


def q(x):
    return 1 + x / 2


def r(x):
    return -np.exp(x / 2)


def f(x):
    return 2 - x


p_x = p(nodes[1:n])
q_x = q(nodes[1:n])
r_x = r(nodes[1:n])
f_x = f(nodes)

# initialize tridiagonal matrix
A = np.zeros((n + 1, 3))
A[1:n, 0] = (-p_x / np.square(h) - q_x / (2 * h))
A[1:n, 1] = -(2 * p_x / np.square(h) + r_x)
A[1:n, 2] = (-p_x / np.square(h) + q_x / (2 * h))
A[0, 1] = -1  # B_0
A[n, 1] = -1  # B_n

f_x[0] = f_x[n] = 0
print(A)


def solve_tdm(A, F):
    s = np.zeros(n + 1)
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    a, b, c = A[:, 0], A[:, 1], A[:, 2]
    g = F
    s[0] = c[0] / b[0]
    t[0] = -g[0] / b[0]
    for i in range(1, n + 1):
        s[i] = c[i] / (b[i] - a[i] * s[i-1])
        t[i] = (a[i] * t[i-1] - g[i]) / (b[i] - a[i]*s[i-1])
    y[n] = t[n]
    for i in range(n-1, -1, -1):
        y[i] = s[i] * y[i+1] + t[i]
    print(s)
    print(t)
    return y


y = solve_tdm(A, f_x)
print(y)


# n = 10
# a, b = -1, 1
# h = (b - a) / n
# nodes = np.arange(a - h / 2, b + h / 2 + h, h)
#
# print(nodes)
#
#
# def p(x):
#     return (2 - x) / (x + 2)
#
#
# def q(x):
#     return x
#
#
# def r(x):
#     return (1 - np.sin(x)) * x
#
#
# def f(x):
#     return np.square(x)
#
#
# p_x = p(nodes[1:n + 1])
# q_x = q(nodes[1:n + 1])
# r_x = r(nodes[1:n + 1])
# f_x = f(nodes)
#
# # initialize tridiagonal matrix
# A = np.zeros((n + 2, 3))
# A[1:n + 1, 0] = (-p_x / np.square(h) - q_x / (2 * h))
# A[1:n + 1, 1] = -(2 * p_x / np.square(h) + r_x)
# A[1:n + 1, 2] = (-p_x / np.square(h) + q_x / (2 * h))
# A[0, 1] = -1 / 2
# A[0, 2] = 1 / 2
# A[n + 1, 0] = 1 / 2
# A[n + 1, 1] = 1 / 2
#
# f_x[0] = f_x[n + 1] = 0
# print(A)
#
#
# def solve_tdm(A, F):
#     s = np.zeros(n + 2)
#     t = np.zeros(n + 2)
#     y = np.zeros(n + 2)
#     a, b, c = A[:, 0], A[:, 1], A[:, 2]
#     g = F
#     s[0] = c[0] / b[0]
#     t[0] = -g[0] / b[0]
#     for i in range(1, n + 2):
#         s[i] = c[i] / (b[i] - a[i] * s[i - 1])
#         t[i] = (a[i] * t[i - 1] - g[i]) / (b[i] - a[i] * s[i - 1])
#     y[n + 1] = t[n + 1]
#     for i in range(n, -1, -1):
#         y[i] = s[i] * y[i + 1] + t[i]
#     return y
#
#
# y = solve_tdm(A, f_x)
# print(y)



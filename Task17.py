from functools import partial
import numpy as np


def a(x, t):
    return 1


def b(x, t):
    return 0


def c(x, t):
    return 0


def alpha1(t):
    return 1


def alpha2(t):
    return 0


def beta1(t):
    return 1


def beta2(t):
    return 0


def explicit_scheme(n, m, t, phi, f, alpha, beta):
    u = np.zeros((m + 1, n + 1))
    h = 1.0 / n
    tau = t / m

    def L(i, k):
        return a(i * h, k * tau) * (u[k][i + 1] - 2 * u[k][i] + u[k][i - 1]) / (h ** 2) + \
               b(i * h, k * tau) * (u[k][i + 1] - u[k][i - 1]) / (2 * h) + c(i * h, k * tau) * u[k][i]

    for i in range(n + 1):
        u[0][i] = phi(i * h)
    for k in range(1, m + 1):
        for i in range(1, n):
            u[k][i] = u[k - 1][i] + tau * (L(i, k - 1) + f(i * h, (k - 1) * tau))
        tk = k * tau
        u[k][0] = (h * alpha(tk) + alpha2(tk) * (2 * u[k][1] - 0.5 * u[k][2])) / \
                  (h * alpha1(tk) + alpha2(tk) * 1.5)
        u[k][n] = (h * beta(tk) + beta2(tk) * (2 * u[k][n - 1] - 0.5 * u[k][n - 2])) / \
                  (h * beta1(tk) + beta2(tk) * 1.5)
    return u


def implicit_scheme(n, m, t, sigma, phi, f, alpha, beta):
    u = np.zeros((m + 1, n + 1))
    h = 1.0 / n
    tau = t / m

    def L(i, k):
        return a(i * h, k * tau) * (u[k][i + 1] - 2 * u[k][i] + u[k][i - 1]) / (h ** 2) + \
               b(i * h, k * tau) * (u[k][i + 1] - u[k][i - 1]) / (2 * h) + c(i * h, k * tau) * u[k][i]

    for i in range(n + 1):
        u[0][i] = phi(i * h)
    for k in range(1, m + 1):
        tk = k * tau - (1 - sigma) * tau
        A = np.zeros(n + 1)
        B = np.zeros(n + 1)
        C = np.zeros(n + 1)
        G = np.zeros(n + 1)
        for i in range(1, n):
            A[i] = sigma * (a(i * h, k * tau) / (h ** 2) - b(i * h, k * tau) / (2 * h))
            B[i] = sigma * (2 * a(i * h, k * tau) / (h ** 2) - c(i * h, k * tau)) - 1 / tau
            C[i] = sigma * (a(i * h, k * tau) / (h ** 2) + b(i * h, k * tau) / (2 * h))
            G[i] = -1.0 * (1 / tau * u[k - 1][i] + (1 - sigma) * L(i, k - 1) + f(i * h, tk))
        B[0] = -(alpha1(k * tau) + alpha2(k * tau) / h)
        C[0] = -alpha2(k * tau) / h
        G[0] = alpha(k * tau)
        A[n] = -beta2(k * tau) / h
        B[n] = -(beta1(k * tau) + beta2(k * tau) / h)
        G[n] = beta(k * tau)
        u[k] = run_matrix(n, A, B, C, G)
    return u


def run_matrix(n, A, B, C, G):
    s = np.zeros(n + 1)
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    s[0] = C[0] / B[0]
    t[0] = -G[0] / B[0]
    for i in range(1, n + 1):
        s[i] = C[i] / (B[i] - A[i] * s[i-1])
        t[i] = (A[i] * t[i-1] - G[i]) / (B[i] - A[i]*s[i-1])
    y[n] = t[n]
    for i in range(n-1, -1, -1):
        y[i] = s[i] * y[i+1] + t[i]
    return y


def get_table(u, N, M):
    table = np.zeros((6, 6))
    h = int(N / 5)
    tay = int(M / 5)
    for i in range(6):
        for j in range(6):
            table[i][j] = u[i * tay][j * h]
    return table


norm_inf = partial(np.linalg.norm, ord=np.inf)


def print_summary(h, tau, dif1, dif2):
    print("h: {h}, tau: {tau}, ||right - u_i||: {dif1}, ||u_i - u_(i-1)||: {dif2} ".format(
        h=h, tau=tau, dif1=dif1, dif2=dif2
    ))

def print_results(correct_solution, sigma, phi, f, alpha, beta):
    right = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            right[i][j] = correct_solution(0.2 * j, 0.02 * i)
    u1 = explicit_scheme(5, 5, 0.1, phi, f, alpha, beta)
    t1 = get_table(u1, 5, 5)
    print("u1: ")
    print(t1)
    u2 = explicit_scheme(10, 20, 0.1, phi, f, alpha, beta)
    t2 = get_table(u2, 10, 20)
    print("u2: ")
    print(t2)
    u3 = explicit_scheme(20, 80, 0.1, phi, f, alpha, beta)
    t3 = get_table(u3, 20, 80)
    print("u3: ")
    print(t3)

    print_summary(0.2, 0.02, norm_inf(right - t1), "-")
    print_summary(0.1, 0.005, norm_inf(right - t2), norm_inf(t2 - t1))
    print_summary(0.05, 0.00125, norm_inf(right - t3), norm_inf(t3 - t2))
    print("-------------------")
    print("Implicit scheme with sigma: ", sigma)
    u1 = implicit_scheme(5, 100, 0.1, sigma, phi, f, alpha, beta)
    t1 = get_table(u1, 5, 5)
    print("u1: ")
    print(t1)
    u2 = implicit_scheme(10, 100, 0.1, sigma, phi, f, alpha, beta)
    t2 = get_table(u2, 10, 20)
    print("u2: ")
    print(t2)
    u3 = implicit_scheme(20, 100, 0.1, sigma, phi, f, alpha, beta)
    t3 = get_table(u3, 20, 80)
    print("u3: ")
    print(t3)
    print_summary(0.2, 0.0001, norm_inf(right - t1), "-")
    print_summary(0.1, 0.0001, norm_inf(right - t2), norm_inf(t2 - t1))
    print_summary(0.05, 0.0001, norm_inf(right - t3), norm_inf(t3 - t2))



sigma = 0.5
print()
print_results(lambda x, t: x, sigma, lambda x: x, lambda x, t: -1, lambda t: 0, lambda t: 1)
print()
print_results(lambda x, t: x ** 3 + t ** 3, sigma, lambda x: x ** 3,
              lambda x, t: 3 * t ** 2 - 6 * x - 3 * x ** 2, lambda t: t ** 3, lambda t: t**3 + 1)

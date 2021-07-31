# Считаем первые 4 узла
    # y.append(y[0] + q(0))
    # y.append(y[1] + (3 * q(1) - q(0)) / 2)
    # y.append(y[2] + (23 * q(2) - 16 * q(1) + 5 * q(0)) / 12)
    # y.append(y[3] + (55 * q(3) - 59 * q(2) + 37 * q(1) - 9 * q(0)) / 24)

# print("Погрешность метода Эйлера: ", abs(precise - euler_method(f, a, b, nodes)))
# print("Погрешность улучшенного метода Эйлера: ", abs(precise - enhanced_euler_method(f, a, b, nodes)))
# print("Погрешность улучшенного метода Эйлера(с формулой трапеций): ")
# print(abs(precise - enhanced_euler_method(f, a, b, nodes, use_trapeze=True)))
# print("Погрешность метода Рунге-Кутта: ", abs(precise - runge_kutta_method(f, a, b, nodes)))
# print("Погрешность экстраполяционного метода Адамса", abs(precise - adams_method(f, a, b, nodes)))

def euler_method(f, a, b, nodes):
    """Uses left-rectangular formula for integral approx"""
    n = len(nodes) - 1
    h = (b - a) / n
    y = [y0]
    for i in range(0, n):
        y.append(y[i] + h * f(nodes[i], y[i]))

    return np.array(y)


def enhanced_euler_method(f, a, b, nodes, use_trapeze=False):
    """Uses either mid-rectangular formula or trapeze formula for integral approx"""
    n = len(nodes) - 1
    h = (b - a) / n
    y = [y0]
    for i in range(0, n):
        if not use_trapeze:
            y.append(y[i] + h * f(nodes[i] + h / 2, y[i] + h * (f(nodes[i], y[i])) / 2))
        else:
            y_hat = y[i] + h * f(nodes[i], y[i])
            y.append(y[i] + h * (f(nodes[i], y[i]) + f(nodes[i+1], y_hat)) / 2)

    return np.array(y)


def runge_kutta_method(f, a, b, nodes):
    """Используем формулу Симпсона"""
    n = len(nodes) - 1
    h = (b - a) / n
    y = [y0]
    for i in range(0, n):
        k1 = h * f(nodes[i], y[i])
        k2 = h * f(nodes[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(nodes[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(nodes[i] + h, y[i] + k3)
        y.append(y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)

    return np.array(y)

import numpy as np

N = 13
h = 0.01  # шаг между узлами
dots_list = [-(N % 3) + i for i in np.arange(0, 1 + h, h)] # пространство точек
dots = np.array(dots_list)


def make_nodes(quantity, x_init, dots):
    h = dots[1] - dots[0]
    m = len(dots) - 1
    if x_init < dots[0] or x_init > dots[m]:
        raise ValueError()

    def get_from_start(quant):
        return dots[:quant]

    def get_from_end(quant):
        return dots[-quant:][::-1]

    # Вычисляем индекс ближайшего узла к исходной точке
    def get_neighbour_index(x_init):
        neighbour = min(dots, key=lambda x: abs(x - x_init))
        return dots.tolist().index(neighbour)

    def check_right_bound(neigh_ind, quant):
        return (neigh_ind + quant // 2) >= m

    def check_left_bound(neigh_ind, quant):
        return (neigh_ind - (quant - 1) // 2) <= 0

    if dots[0] < x_init <= dots[0] + h / 2:
        result = get_from_start(quantity)
        ind = 0
    elif dots[m] - h / 2 <= x_init < dots[m]:
        result = get_from_end(quantity)
        ind = quantity - 1
    else:
        n_ind = get_neighbour_index(x_init)
        if check_left_bound(n_ind, quantity):
            result = get_from_start(quantity)
            ind = 0
        elif check_right_bound(n_ind, quantity):
            result = get_from_end(quantity)
            ind = quantity - 1
        else:
            result = dots[n_ind - (quantity - 1) // 2:n_ind + (quantity // 2) + 1]
            ind = result.tolist().index(dots[n_ind])
    return result, ind


print(make_nodes(5, -0.5, dots))
print(make_nodes(5, -1, dots))
print(make_nodes(5, 0, dots))
print(make_nodes(5, -0.999, dots))
print(make_nodes(5, -0.001, dots))
print(make_nodes(4, -0.017, dots))
print(make_nodes(10, -1, dots))

""" Output:
50 -0.5
3.3
(array([-0.52, -0.51, -0.5 , -0.49, -0.48]), 50)
3.1
(array([-1.  , -0.99, -0.98, -0.97, -0.96]), 0)
3.2
(array([ 0.  , -0.01, -0.02, -0.03, -0.04]), 100)
1
(array([-1.  , -0.99, -0.98, -0.97, -0.96]), 0)
2
(array([ 0.  , -0.01, -0.02, -0.03, -0.04]), 100)
3.2
(array([ 0.  , -0.01, -0.02, -0.03]), 100)
3.1
(array([-1.  , -0.99, -0.98, -0.97, -0.96, -0.95, -0.94, -0.93, -0.92,
       -0.91]), 0)
"""

def newton_equal_dist1(f, nodes, start_ind):
    n = len(nodes) - 1
    h = nodes[0] - nodes[1]
    result_t = Polynomial(f(nodes[start_ind]))

    if start_ind == 0:
        def coef_func(x): return -x
    elif start_ind == len(dots) - 1:
        def coef_func(x): return x
    else:
        def coef_func(x): return ((-1) ** x) * ((x + 1) // 2)

    for i in range(n):
        sum_part = n_k(i, coef_func) * finite_difference(f, nodes[start_ind], i + 1, h)
        result_t = pl.polyadd(result_t, sum_part)[0]

    x, t = Symbol('x'), Symbol('t')
    # x = nodes[start_ind] + t * h
    t = (x - nodes[start_ind]) / h
    print(result_t)
    result_t = sp.Poly.from_list(result_t.coef, gens=t)
    print(result_t)

    result = sp.simplify(result_t)
    print(result)

    return result.all_coeffs()

def newton_equal_dist(f, nodes, start_ind):
    n = len(nodes) - 1
    h = nodes[1] - nodes[0]
    result_t = Polynomial(f(nodes[start_ind]))
    dm = diff_matrix(f, nodes)

    # Разбор случаев расположения исходной точки относительно узлов
    if start_ind == 0:
        flag = 0
        def coef_func(x): return -x
    elif start_ind == len(nodes) - 1:
        flag = 1
        def coef_func(x): return x
    else:
        flag = 2
        def coef_func(x): return ((-1) ** x) * ((x + 1) // 2)

    fin_dif_list = process_diff_matrix(dm, flag, start_ind)

    for i in range(n):
        sum_part = n_k(i, coef_func) * fin_dif_list[i + 1]
        result_t = pl.polyadd(result_t, sum_part)[0]

    x, t = Symbol('x'), Symbol('t')
    # x = nodes[start_ind] + t * h
    t = (x - nodes[start_ind]) / h
    # print(result_t)
    result_t = sp.Poly.from_list(result_t.coef[::-1], gens=t)
    # print(result_t)

    result = sp.simplify(result_t)

    return Polynomial(result.all_coeffs()[::-1])


from matplotlib import pyplot as plt
import numpy as np

plt.rcParams["patch.force_edgecolor"] = True
# a, b = 0, 2 * np.pi
z = 1  # точка, в которой считаем производную


def f(x):
    return np.power(x, 1 / 3) * np.sin(x)


def df(x):
    return (np.power(x, -2 / 3) / 3) * np.sin(x) + np.power(x, 1 / 3) * np.cos(x)


def df2(x):
    return -2 * np.sin(x) * np.power(x, -5 / 3) / 9 + 2 * np.cos(x) * np.power(x, -2 / 3) / 3 - \
           np.power(x, 1 / 3) * np.sin(x)


def df_num_right(h):
    return (f(z + h) - f(z)) / h


def df_num_left(h):
    return (f(z + h) - f(z)) / h


def df_num_mid1(h):
    return (f(z + h) - f(z - h)) / (2 * h)


def df_num_mid2(h):
    return (-3 * f(z) + 4 * f(z + h) - f(z + 2 * h)) / (2 * h)


def df_num_mid3(h):
    return (3 * f(z) - 4 * f(z - h) + f(z - 2 * h)) / (2 * h)


def df2_num(h):
    return (f(z + h) - 2 * f(z) + f(z - h)) / (h ** 2)


h = 0.1

print('"Точное" значение производной первого порядка')
print(df(z))

print("Приближённые производные первого порядка и их разность с точной")
print(df_num_left(h), abs(df_num_left(h) - df(z)))
print(df_num_right(h), abs(df_num_right(h) - df(z)))
print(df_num_mid1(h), abs(df_num_mid1(h) - df(z)))
print(df_num_mid2(h), abs(df_num_mid2(h) - df(z)))
print(df_num_mid3(h), abs(df_num_mid3(h) - df(z)))

print('"Точное" значение производной второго порядка')
print(df2(z))

print("Приближённая производная второго порядка и её разность с точной")
print(df2_num(h), abs(df2_num(h) - df2(z)))

print('Найдём оптимальный шаг для второй производной')
min_h = 0.01
min_diff = 1.0

for h in np.arange(1e-5, 1e-3, 1e-6):
    diff = abs(df2_num(h) - df2(z))
    # print(diff)
    if diff < min_diff:
        min_diff = diff
        min_h = h
# h [0.00010; 0.00014]
# print('Оптимальный шаг лежит в промежутке [0.00010; 0.00014] ')

print('Найденное оптимальное значение h:', min_h)
print('Погрешность при оптимальном шаге:', abs(df2_num(min_h) - df2(z)))
print('Погрешность при большем шаге:', abs(df2_num(1e-3) - df2(z)))
print('Погрешность при меньшем шаге:', abs(df2_num(1e-5) - df2(z)))

x1 = np.linspace(1e-5, 1e-3, 100)
plt.plot(x1, abs(df2_num(x1) - df2(z)))
plt.plot(min_h, df2_num(min_h) - df2(z), 'o')
plt.show()



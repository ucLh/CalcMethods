
# coding: utf-8

# In[66]:


from functools import partial
from matplotlib import pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as pl
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import minimize
from scipy.special import factorial
import sympy as sp
from sympy import Symbol, diff
from sympy.utilities import lambdify

plt.rcParams["patch.force_edgecolor"] = True
N = 13
x = np.array([-(N % 3), -(N % 3) + 0.1, -(N % 3) + 0.3, -(N % 3) + 0.45, -(N % 3) + 0.5])
n = len(x)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[67]:


def f_gen(x, n):
    return x * np.exp(x * ((n % 2) + 1)) + np.sin(x * ((n % 7) + 1) / 2)

def f_gen_sympy(x, n):
    return x * sp.exp(x * ((n % 2) + 1)) + sp.sin(x * ((n % 7) + 1) / 2)
# Делаем две функции, для двух библиотек: sympy для вычисления производной, numpy, чтобы было удобно 
# считать функцию сразу от целого массива точек


# In[68]:


# Функции для вычисления интерпол. полинома в форме Лагранжа
def lagrange(nodes):
    result = Polynomial([0])
    # np.poly строит полином по набору корней. Но порядок коэффициетов противоположный тому, который
    # принимает конструктор класса Polynomial
    w = Polynomial(np.poly(nodes)[::-1])
    deriv = w.deriv(1)
    for i in range(len(nodes)):
        result = pl.polyadd(result, make_l_k(i, nodes, w, deriv) * f(nodes[i]))[0] 
        # возвращается не только сумма
    return result

def make_l_k(k, nodes, w, deriv):
    poly_part = np.polydiv(w.coef, np.array([-nodes[k], 1]))[0]
    return Polynomial(poly_part) / deriv(nodes[k])


# In[69]:


# Функции для вычисления теоретической погрешности
def get_coef(f, c, d):
    y = Symbol('y')
    # Дифференцируем 5 раз
    deriv = diff(f(y), y, y, y, y, y)
    # Получаем функцию из объекта библиотеки sympy
    deriv_func = lambdify(y, -abs(deriv))
    # Минимизируем отрицательную производную, задав начальную точку и интервал поиска
    max_deriv = -minimize(deriv_func, [np.mean([c, d])], bounds=[(c, d)]).fun[0]
    return max_deriv / factorial(n)


def max_error(x, nodes, coef):
    w = Polynomial(np.poly(nodes)[::-1])
    return coef * abs(w(x))


# In[70]:


def newton_interpol_polynomial(nodes, f):
    n = len(nodes)
    dif_matrix = np.zeros((n, n))

    for i in range(n):
        dif_matrix[i, 0] = f(nodes[i])
    
    # Считаем разделённые разности. Храним их в верхнетреугольной матрице
    for i in range(1, n):
        for j in range(0, n - i):
            dif_matrix[j, i] = (dif_matrix[j + 1, i - 1] - dif_matrix[j, i - 1]) / (nodes[j + i] - nodes[j])

    # Строим интерпол. полином Ньютона, испоьзуя полученные разности
    result = Polynomial(dif_matrix[0, 0])

    for i in range(n - 1):
        result = pl.polyadd(result, Polynomial(np.poly(nodes[0:i + 1])[::-1]) * dif_matrix[0, i + 1])[0]

    # Помимо полинома возвращаем ещё и матрицу разностей, т.к. её, вообще говоря, можно использовать,
    # если мы захотим добавить ещё узлов интерполяции
    return result, dif_matrix


# In[71]:


def chebyshev_roots(n, c, d):
    roots = []
    for i in range(n + 1):
        roots.append(np.cos(((2 * i + 1) * np.pi) / (2 * n + 2)))
    return list(map(lambda x: x * (d - c) / 2 + (d + c) / 2, roots))


# In[72]:


# Задаем константу N у функции, в моем случае N = 13
f = partial(f_gen, n=N)
f_sp = partial(f_gen_sympy, n=N)


# In[73]:


p_lagr = lagrange(x)
print(p_lagr.coef, type(p_lagr)) # Подтверждаем, что нашли полином Лагранжа в аналитической форме


# In[74]:


x1 = np.linspace(x[0], x[n - 1])
plt.plot(x1, f(x1), label='f(x)')
plt.plot(x1, p_lagr(x1), label='p_n(x)')
plt.plot(x, f(x), 'o')
plt.legend(loc='upper left')
plt.title('Approximation with Lagrange polynomial')


# In[75]:


# Оцениваем теоретическую и фактическую погрешности
coef = get_coef(f_sp, x[0], x[n - 1])
plt.plot(x1, abs(f(x1) - p_lagr(x1)), label='fact_error')
plt.plot(x1, max_error(x1, x, coef), label='theor_error')
plt.legend(loc='upper left')
plt.title('Errors using fixed nodes')
# Замечаем, что фактическая ошибка действительно меньше теоретической


# In[76]:


p_newton, _ = newton_interpol_polynomial(x, f)
print(p_newton.coef)
print(p_lagr.coef)
# Видим, что интерпол. многочлены совпадают, что ожидаемо, по единственности интерпол. многочлена


# In[77]:


# Считаем корни Чебышева для гашего отрезка
x_cheb = chebyshev_roots(n - 1, x[0], x[n - 1])
p_cheb, _ = newton_interpol_polynomial(x_cheb, f)


# In[78]:


plt.plot(x1, abs(f(x1) - p_cheb(x1)), label='fact_error')
plt.plot(x1, max_error(x1, x_cheb, coef), label='theor_error')
plt.legend(loc='upper left')
plt.title('Errors using Chebyshev nodes')


# Как мы видим, как теоретическая, так и фактическая ошибка уменьшаются при использовании в качестве узлов интерполяции корней Чебышёва

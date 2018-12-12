import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math as m


def f(x_, a):
    # return x_**8 + x_**2 + x_
    return x_ * m.log(x_ + a + 0.1)


def f_derivative(x_, a):
    x = sp.symbols('x')
    # y = x**8 + x**2 + x
    y = x * sp.log(x + a + 0.1)
    return y.diff(x).subs('x', x_)


def plotting(x, y, y_new, x_knots, y_knots, format_1, format_2, format_3, title):
    plt.plot(x, y, format_1, x, y_new, format_2)
    plt.plot(x_knots, y_knots, format_3)
    plt.title(title)
    plt.grid(True)
    plt.show()


def chebyshev(i, a, n):
    return a * m.cos(m.pi * (2 * i + 1) / (2 * (n + 1)))


def system(x_, y_, a_, n):
    a_matrix = np.full((3*n - 3, 3*n - 3), 0.0)
    for i in range(0, 2*n-2, 2):
        for j in range(3):
            a_matrix[i][(3*i//2)+j] = x_[i//2]**(2 - j)
            a_matrix[i + 1][(3*i//2) + j] = x_[i//2 + 1] ** (2 - j)
    for i in range(2*n-2, 3*n-4):
        for j in range(2):
            a_matrix[i][3*(i - 2*n + 2) + 3*j] = (1 - 2*j)*2*x_[i - 2*n + 3]
            a_matrix[i][3*(i - 2*n + 2) + 3*j + 1] = 1 - 2*j
    a_matrix[3*n - 4, 0] = x_[0]*2
    a_matrix[3 * n - 4, 1] = 1
    y_vector = np.full((3*n - 3, 1), 0.0)
    for i in range(n):
        if i != 0 and i != n-1:
            y_vector[2*i - 1][0] = y_[i]
            y_vector[2*i][0] = y_[i]
        elif i == 0:
            y_vector[i][0] = y_[i]
        else:
            y_vector[2*i-1][0] = y_[i]
    # y_vector[3*n - 4][0] = f_derivative(x_[0], a_)
    y_vector[3 * n - 4][0] = -1
    a = np.linalg.solve(a_matrix, y_vector)
    a = list(np.ravel(a))
    return a


def splines(x_knots, y_knots, x_, a_, n):
    a = system(x_knots, y_knots, a_, n)
    y_ = []
    print(a)
    for i, point in enumerate(x_):
        k = i//200
        y_.append(a[3*k]*point**2 + a[3*k+1]*point + a[3*k+2])
    return y_


def main(a, n, func):
    x = np.linspace(-a, a, (n-1)*200)
    y = [func(i, a) for i in x]
    x_knots = np.linspace(-a, a, n)
    y_knots = [func(i, a) for i in x_knots]
    y_s = splines(x_knots, y_knots, x, a, n)
    plotting(x, y, y_s, x_knots, y_knots, 'b', 'g', 'c.', 'Quadratic Splines')


main(5, 3, f)

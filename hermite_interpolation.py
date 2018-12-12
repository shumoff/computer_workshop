import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math


def f(x_, a):
    # return x_**8 + x_**2 + x_
    return x_ * math.log(x_ + a + 0.1)


def f_derivative(x_, m, a):
    x = sp.symbols('x')
    y = x * sp.log(x + a + 0.1)
    # y = x**8 + x**2 + x
    return y.diff(x, m).subs('x', x_)


def vandermonde(x_knots, n):
    q_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            q_matrix[i][j] = x_knots[i]**j
    return q_matrix


def canonical_coefficients(x_knots, y_knots, n):
    y_vector = np.matrix([y_knots[i] for i in range(n)]).transpose()
    w_matrix = vandermonde(x_knots, n)
    a_ = np.linalg.solve(w_matrix, y_vector)
    a_ = list(np.ravel(a_))
    a_.reverse()
    return a_


def plotting(x, y, y_new, x_knots, y_knots, format_1, format_2, format_3, title):
    plt.plot(x, y, format_1, x, y_new, format_2)
    plt.plot(x_knots, y_knots, format_3)
    plt.title(title)
    plt.grid(True)
    plt.show()


def copying_points(x_, y_, m):
    z = [0]*((m+1)*len(x_))
    f_z = [0]*(m+1)*len(x_)
    for i in range(len(x_)):
        for j in range(m+1):
            z[(m+1)*i + j] = x_[i]
            f_z[(m+1)*i + j] = y_[i]
    return [z, f_z]


def div_diffs_table(z, f_z, m_, a_):
    table = np.zeros((len(z), len(z)))
    for j in range(len(z)):
        for i in range(len(z)):
            if (i+j)//(m_+1) == i//(m_+1):
                table[i][j] = f_derivative(z[i], j, a_) / math.factorial(j)
            elif j != 0 and i+j < len(z):
                table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (z[j+i] - z[i])
            elif j != 0:
                pass
            else:
                table[i][j] = f_z[i]
    print(table)
    return table


def hermite(x_knots, y_knots, x_, m_, a_):
    z, f_z = copying_points(x_knots, y_knots, m_)
    y_ = []
    polynomial = 0
    multiplication = 1
    table = div_diffs_table(z, f_z, m_, a_)
    for point in x_:
        for j in range(len(z)):
            for i in range(j):
                multiplication *= point - z[i]
            polynomial += multiplication * table[0][j]
            multiplication = 1
        y_.append(polynomial)
        polynomial = 0
    return y_


def sym_hermite(x_knots, y_knots, m_, a_):
    z, f_z = copying_points(x_knots, y_knots, m_)
    x_ = sp.symbols('x')
    polynomial = 0
    multiplication = 1
    table = div_diffs_table(z, f_z, m_, a_)
    for j in range(len(z)):
        for i in range(j):
            multiplication *= x_ - z[i]
        polynomial += multiplication * table[0][j]
        multiplication = 1
    return polynomial


def main(a, n, m, func):
    x = np.linspace(-a, a, (n-1)*200)
    y = [func(i, a) for i in x]
    x_knots = np.linspace(-a, a, n)
    y_knots = [func(i, a) for i in x_knots]
    test_x_knots = np.linspace(-a, a, n*(m+1))
    test_y_knots = [func(i, a) for i in test_x_knots]
    y_h = hermite(x_knots, y_knots, x, m, a)
    plotting(x, y, y_h, x_knots, y_knots, 'b', 'g', 'c.', 'Hermite Interpolation')
    print('Hermite polynomial: ', sp.simplify(sym_hermite(x_knots, y_knots, m, a)))
    print('Canonical coefficients: ', canonical_coefficients(test_x_knots, test_y_knots, n*(m+1)))


main(5, 10, 2, f)

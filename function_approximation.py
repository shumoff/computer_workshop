import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math as m


def f(x_, a):
    return x_ * m.tan(x_ + a + 0.1)
    # return x_ * m.log(x_ + a + 0.1)


def plotting(x, y, y_new, x_knots, y_knots, format_1, format_2, format_3, title):
    plt.plot(x, y, format_1, x, y_new, format_2)
    plt.plot(x_knots, y_knots, format_3)
    plt.title(title)
    plt.grid(True)
    plt.show()


def chebyshev(i, a, n):
    return a * m.cos(m.pi * (2 * i + 1) / (2 * (n + 1)))


def canonical_coefficients(x_, y_, n):
    y_vector = np.matrix([y_[i] for i in range(n)]).transpose()
    w_matrix = q(x_, n)
    a_ = np.linalg.solve(w_matrix, y_vector)
    a_ = list(np.ravel(a_))
    a_.reverse()
    return a_


def q(x_, n):
    q_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            q_matrix[i][j] = x_[i]**j
    return q_matrix


def solution(x_, y_, n):
    h_matrix = q(x_, n).transpose()
    y_vector = np.matrix([y_[i] for i in range(n)]).transpose()
    b_vector = h_matrix * y_vector
    h_matrix = np.matmul(h_matrix, h_matrix.transpose())
    a_ = np.linalg.solve(h_matrix, b_vector)
    a_ = list(np.ravel(a_))
    return a_


def polynomial(x_, a_, n):
    p = 0
    for i in range(n):
        p += a_[i]*x_**i
    return p


def sym_polynomial(a_, n):
    p = 0
    x_ = sp.symbols('x')
    for i in range(n):
        p += a_[i]*x_**i
    return p


def main(a, n, func):
    x_new = np.linspace(-a, a, 1000)
    y_f = [func(i, a) for i in x_new]
    x = np.linspace(-a, a, n)
    y = [func(i, a) for i in x]
    x_ch = [chebyshev(i, a, n) for i in range(n)]
    y_ch = [func(i, a) for i in x_ch]
    a_ = solution(x, y, n)
    a_ch = solution(x_ch, y_ch, n)
    p = [polynomial(point, a_, n) for point in x_new]
    p_ch = [polynomial(point, a_ch, n) for point in x_new]
    plotting(x_new, y_f, p, x, y, 'b', 'g', 'co', 'Least square method')
    plotting(x_new, y_f, p_ch, x_ch, y_ch, 'b', 'r', 'mo', 'Least square method (Chebyshev)')
    print('Approximation polynomial: ', sym_polynomial(a_, n))
    print('Canonical coefficients: ', canonical_coefficients(x, y, n))
    print('Approximation polynomial (Chebyshev): ', sym_polynomial(a_ch, n))
    print('Canonical coefficients (Chebyshev): ', canonical_coefficients(x_ch, y_ch, n))


main(0.7, 6, f)

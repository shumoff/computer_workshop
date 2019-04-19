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


def canonical_coefficients(x_knots, y_knots, n):
    y_vector = np.matrix([y_knots[i] for i in range(n)]).transpose()
    w_matrix = q(x_knots, n, n)
    a_ = np.linalg.solve(w_matrix, y_vector)
    a_ = list(np.ravel(a_))
    a_.reverse()
    return a_


def q(x_knots, n, m_):
    q_matrix = np.zeros((n, m_))
    for i in range(n):
        for j in range(m_):
            q_matrix[i][j] = x_knots[i] ** j
    return q_matrix


def solution(x_knots, y_knots, n, m_):
    h_matrix = q(x_knots, n, m_).transpose()
    y_vector = np.matrix([y_knots[i] for i in range(n)]).transpose()
    b_vector = h_matrix * y_vector
    h_matrix = np.matmul(h_matrix, h_matrix.transpose())
    a_ = np.linalg.solve(h_matrix, b_vector)
    a_ = list(np.ravel(a_))
    return a_


def polynomial(x_knots, a_, m_):
    p = 0
    for i in range(m_):
        p += a_[i] * x_knots ** i
    return p


def sym_polynomial(a_, m_):
    p = 0
    x_ = sp.symbols('x')
    for i in range(m_):
        p += a_[i]*x_**i
    return p


def main(a, n, m_, func):
    x = np.linspace(-a, a, (n-1)*200)
    y = [func(i, a) for i in x]
    x_knots = np.linspace(-a, a, n)
    y_knots = [func(i, a) for i in x_knots]
    x_ch_knots = [chebyshev(i, a, n) for i in range(n)]
    y_ch_knots = [func(i, a) for i in x_ch_knots]
    test_x_knots = np.linspace(-a, a, m_)
    test_y_knots = [func(i, a) for i in test_x_knots]
    test_x_ch_knots = [chebyshev(i, a, n) for i in range(m_)]
    test_y_ch_knots = [func(i, a) for i in test_x_ch_knots]
    a_ = solution(x_knots, y_knots, n, m_)
    a_ch = solution(x_ch_knots, y_ch_knots, n, m_)
    p = [polynomial(point, a_, m_) for point in x]
    p_ch = [polynomial(point, a_ch, m_) for point in x]
    plotting(x, y, p, x_knots, y_knots, 'b', 'g', 'co', 'Least squares method')
    plotting(x, y, p_ch, x_ch_knots, y_ch_knots, 'b', 'r', 'mo', 'Least squares method (Chebyshev)')
    print('Approximation polynomial (least squares method): ', sym_polynomial(a_, m_))
    print('Canonical coefficients: ', canonical_coefficients(test_x_knots, test_y_knots, m_))
    print('Approximation polynomial (least squares method, Chebyshev): ', sym_polynomial(a_ch, m_))
    print('Canonical coefficients (Chebyshev): ', canonical_coefficients(test_x_ch_knots, test_y_ch_knots, m_))


if __name__ == "__main__":
    main(0.7, 5, 3, f)

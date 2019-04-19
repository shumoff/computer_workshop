import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math as m


def f(x_, a):
    return x_ * m.cos(x_ + a + 0.1)


def f_module(x_, a):
    return abs(x_) * x_ * m.log(x_ + a + 0.1)


def plotting(x, y, y_new, x_knots, y_knots, format_1, format_2, format_3, title):
    plt.plot(x, y, format_1, x, y_new, format_2)
    plt.plot(x_knots, y_knots, format_3)
    plt.title(title)
    plt.grid(True)
    plt.show()


def chebyshev(i, a, n):
    return a * m.cos(m.pi * (2 * i + 1) / (2 * (n + 1)))


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


def lagrange(x_knots, y_knots, point, n):
    polynomial = 0
    for j in range(n):
        lagrange_multiplier = 1
        for i in range(n):
            if i != j:
                lagrange_multiplier *= (point - x_knots[i]) / (x_knots[j] - x_knots[i])
        polynomial += y_knots[j] * lagrange_multiplier
    return polynomial


def l_polynomial(x_knots, y_knots, n):
    sym_polynomial = 0
    w = sp.symbols('x')
    for j in range(n):
        sym_lagrange_multiplier = 1
        for i in range(n):
            if i != j:
                sym_lagrange_multiplier *= (w - x_knots[i]) / (x_knots[j] - x_knots[i])
        sym_polynomial += y_knots[j] * sym_lagrange_multiplier
    return sym_polynomial


def coefficient(x_knots, y_knots, k, coefficients):
    c = 0
    differences = 1
    for i in range(k):
        differences *= (x_knots[k] - x_knots[k - i - 1])
        c -= coefficients[k-i-1] / differences
    c += y_knots[k] / differences
    coefficients[k] = c
    return c


def newton(x_knots, y_knots, point, n, coefficients):
    polynomial = 0
    for j in range(n):
        differences = 1
        for i in range(j):
            differences *= (point - x_knots[i])
        polynomial += coefficient(x_knots, y_knots, j, coefficients) * differences
    return polynomial


def n_polynomial(x_knots, y_knots, n, coefficients, sym=False):
    sym_polynomial = 0
    w = sp.symbols('x')
    for j in range(n):
        difference = 1
        for i in range(j):
            difference *= (w - x_knots[i])
        if sym:
            sym_c = sp.symbols('c{}'.format(j))
            sym_polynomial += sym_c * difference
        else:
            sym_polynomial += coefficient(x_knots, y_knots, j, coefficients) * difference
    return sym_polynomial


def main(a, n, func):
    coefficients = [0]*n
    x = np.linspace(-a, a, (n-1)*200)
    y = [func(i, a) for i in x]
    x_knots = np.linspace(-a, a, n)
    y_knots = [func(i, a) for i in x_knots]
    x_ch_knots = [chebyshev(i, a, n) for i in range(n)]
    y_ch_knots = [func(i, a) for i in x_ch_knots]
    y_l = [lagrange(x_knots, y_knots, point, n) for point in x]
    y_n = [newton(x_knots, y_knots, point, n, coefficients) for point in x]
    y_ch_l = [lagrange(x_ch_knots, y_ch_knots, point, n) for point in x]
    y_ch_n = [newton(x_ch_knots, y_ch_knots, point, n, coefficients) for point in x]
    plotting(x, y, y_l, x_knots, y_knots, 'b', 'g', 'co', 'Lagrange')
    plotting(x, y, y_n, x_knots, y_knots, 'b', 'g', 'co', 'Newton')
    plotting(x, y, y_ch_l, x_ch_knots, y_ch_knots, 'b', 'r', 'mo', 'Lagrange (Chebyshev)')
    plotting(x, y, y_ch_n, x_ch_knots, y_ch_knots, 'b', 'r', 'mo', 'Newton (Chebyshev)')
    print('Lagrange polynomial: ', sp.simplify(l_polynomial(x_knots, y_knots, n)))
    print('Newton polynomial:', sp.simplify(n_polynomial(x_knots, y_knots, n, coefficients)))
    print('Canonical coefficients: ', canonical_coefficients(x_knots, y_knots, n))
    print('Lagrange polynomial (Chebyshev): ', sp.simplify(l_polynomial(x_ch_knots, y_ch_knots, n)))
    print('Newton polynomial (Chebyshev):', sp.simplify(n_polynomial(x_ch_knots, y_ch_knots, n, coefficients)))
    print('Canonical coefficients (Chebyshev): ', canonical_coefficients(x_ch_knots, y_ch_knots, n))


if __name__ == "__main__":
    main(5, 7, f)

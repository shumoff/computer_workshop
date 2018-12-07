import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math as m


def f(x_, a):
    return x_ * m.log(x_ + a + 0.1)


def f_module(x_, a):
    return abs(x_) * x_ * m.log(x_ + a + 0.1)


def chebyshev(i, a, n):
    return a * m.cos(m.pi * (2 * i + 1) / (2 * (n + 1)))


def vandermonde(x_, n):
    q_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            q_matrix[i][j] = x_[i]**j
    return q_matrix


def canonical_coefficients(x_, y_, n):
    y_vector = np.matrix([y_[i] for i in range(n)]).transpose()
    w_matrix = vandermonde(x_, n)
    a_ = np.linalg.solve(w_matrix, y_vector)
    a_ = list(np.ravel(a_))
    a_.reverse()
    return a_


def lagrange(x_, y_, point, n):
    polynomial = 0
    for j in range(n):
        lagrange_multiplier = 1
        for i in range(n):
            if i != j:
                lagrange_multiplier *= (point - x_[i])/(x_[j] - x_[i])
        polynomial += y_[j] * lagrange_multiplier
    return polynomial


def l_polynomial(x_, y_, n):
    sym_polynomial = 0
    w = sp.symbols('x')
    for j in range(n):
        sym_lagrange_multiplier = 1
        for i in range(n):
            if i != j:
                sym_lagrange_multiplier *= (w - x_[i]) / (x_[j] - x_[i])
        sym_polynomial += y_[j] * sym_lagrange_multiplier
    return sym_polynomial


def coefficient(x_, y_, k, coefficients):
    c = 0
    differences = 1
    for i in range(k):
        differences *= (x_[k] - x_[k-i-1])
        c -= coefficients[k-i-1] / differences
    c += y_[k]/differences
    coefficients[k] = c
    return c


def newton(x_, y_, point, n, coefficients):
    polynomial = 0
    for j in range(n):
        differences = 1
        for i in range(j):
            differences *= (point - x_[i])
        polynomial += coefficient(x_, y_, j, coefficients) * differences
    return polynomial


def n_polynomial(x_, y_, n, coefficients, sym=False):
    sym_polynomial = 0
    w = sp.symbols('x')
    for j in range(n):
        difference = 1
        for i in range(j):
            difference *= (w - x_[i])
        if sym:
            sym_c = sp.symbols('c{}'.format(j))
            sym_polynomial += sym_c * difference
        else:
            sym_polynomial += coefficient(x_, y_, j, coefficients) * difference
    return sym_polynomial


def plotting(a, n, func):
    coefficients = [0]*n
    x_new = np.linspace(-a, a, 1000)
    y_f = [func(i, a) for i in x_new]
    x = np.linspace(-a, a, n)
    y = [func(i, a) for i in x]
    x_ch = [chebyshev(i, a, n) for i in range(n)]
    y_ch = [func(i, a) for i in x_ch]
    y_l = [lagrange(x, y, point, n) for point in x_new]
    y_n = [newton(x, y, point, n, coefficients) for point in x_new]
    y_ch_l = [lagrange(x_ch, y_ch, point, n) for point in x_new]
    y_ch_n = [newton(x_ch, y_ch, point, n, coefficients) for point in x_new]
    # c = [abs(y_f[i] - y_l[i]) for i in range(n)]
    # print(max(c))
    plt.plot(x_new, y_f, 'b', x_new, y_l, 'g', x_new, y_ch_l, 'r')
    plt.plot(x_new, y_f, 'b', x_new, y_n, 'g', x_new, y_ch_n, 'r')
    print('Lagrange polynomial: ', sp.simplify(l_polynomial(x, y, n)), '\n')
    print('Lagrange polynomial (Chebyshev): ', sp.simplify(l_polynomial(x_ch, y_ch, n)), '\n')
    print('Newton polynomial:', sp.simplify(n_polynomial(x, y, n, coefficients)), '\n')
    print('Newton polynomial (Chebyshev):', sp.simplify(n_polynomial(x_ch, y_ch, n, coefficients)), '\n')
    print('Canonical coefficients: ', canonical_coefficients(x, y, n))
    print('Canonical coefficients (Chebyshev): ', canonical_coefficients(x_ch, y_ch, n))
    plt.grid(True)
    plt.show()


plotting(5, 4, f)

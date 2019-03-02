import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math


def f(x_, a):
    # return x_ * math.tan(x_ + a + 0.1)
    return x_ * math.log(x_ + a + 0.1)
    # return x_ ** 8 + x_ ** 2 + x_


def derivative(expr, m):
    x = sp.symbols('x')
    y = expr
    # y = x**8 + x**2 + x
    return y.diff(x, m)


def plotting(x, y, y_new, x_knots, y_knots, format_1, format_2, format_3, title):
    plt.plot(x, y, format_1, x, y_new, format_2)
    plt.plot(x_knots, y_knots, format_3)
    plt.title(title)
    plt.grid(True)
    plt.show()


def canonical_coefficients(x_knots, y_knots, n):
    y_vector = np.matrix([y_knots[i] for i in range(n)]).transpose()
    w_matrix = vandermonde(x_knots, n, n)
    a_ = np.linalg.solve(w_matrix, y_vector)
    a_ = list(np.ravel(a_))
    a_.reverse()
    return a_


def vandermonde(x_knots, n, m_):
    q_matrix = np.zeros((n, m_))
    for i in range(n):
        for j in range(m_):
            q_matrix[i][j] = x_knots[i] ** j
    return q_matrix


def legendre_polynomials(x, a, n):
    polynomial = 0
    x_ = sp. symbols('x')
    y_ = []
    expr_1 = (1 - x_ ** 2) ** n
    for point in x:
        for i in range(n):
            l = sp.diff(expr_1, x_, n) / (math.factorial(i)*2**i)
            c = sp.integrate(x_ * sp.log(x_ + a + 0.1)*l, (x_, -a, a)) / sp.integrate(l**2, (x_, -a, a))
            l = l.subs(x_, point)
            polynomial += c*l
        y_.append(polynomial)
        polynomial = 0
    return y_


def main(a, n, func):
    x = np.linspace(-a, a, (n-1)*10)
    y = [func(i, a) for i in x]
    x_knots = np.linspace(-a, a, n)
    y_knots = [func(i, a) for i in x_knots]
    y_l = legendre_polynomials(x, a, n)
    plotting(x, y, y_l, x_knots, y_knots, 'b', 'g', 'c.', 'Legendre Polynomials')


main(1, 5, f)


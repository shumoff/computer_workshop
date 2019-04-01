import math
import time
import numpy as np
import sympy as sp
from scipy import integrate, optimize

epsilon_ = 10 ** -6
a_ = 1.8
b_ = 2.9
n_ = 3
alpha = 0
beta = 4/7


def function(x, opt=False, diff=False):
    if opt:
        return -abs(4 * math.cos(2.5*x) * math.exp(4*x/7) + 2.5 * math.sin(5.5*x) * math.exp(-3*x/5) + 4.3 * x)
    elif diff:
        x_ = sp.Symbol("x")
        return 4 * sp.cos(2.5 * x_) * sp.exp(4 * x_ / 7) + 2.5 * sp.sin(5.5 * x_) * sp.exp(-3 * x_ / 5) + 4.3 * x_
    return 4 * math.cos(2.5*x) * math.exp(4*x/7) + 2.5 * math.sin(5.5*x) * math.exp(-3*x/5) + 4.3 * x


def weight_func(x, j):
    return ((x - a_)**-alpha) * ((b_-x)**-beta) * (x**j)


def knot_polynomial(x, knots, j=-1):
    res = 1
    for i in range(len(knots)):
        if i != j:
            res *= x - knots[i]
    return res


def wanted_function(x):
    return function(x) * weight_func(x, 0)


def knot_polynomial_derivative(x, knots):
    res = 0
    for i in range(len(knots)):
        trans_res = 1
        for j in range(len(knots)):
            if j != i:
                trans_res *= x - knots[j]
        res += trans_res
    return res


def estimator_func(x, knots):
    return abs(weight_func(x, 0) * knot_polynomial(x, knots))


def coefficients_func(x, knots, j):
    return weight_func(x, 0) * knot_polynomial(x, knots, j) / knot_polynomial_derivative(knots[j], knots)


def coefficients(a, b, knots):
    coefficients_ = [k[0] for k in [integrate.quad(coefficients_func, a, b, args=(knots, j))
                                    for j in range(len(knots))]]
    return coefficients_


def regular_iqr(a, b, n):
    knots = np.linspace(a, b, n)
    moments = [k[0] for k in [integrate.quad(weight_func, a, b, args=(j,)) for j in range(n)]]
    knots_matrix = np.zeros((n, n))
    for s in range(n):
        for j in range(n):
            knots_matrix[s][j] = knots[j] ** s
    moments_vector = np.matrix(moments).transpose()
    quadrature_coefficients = list(np.ravel(np.linalg.solve(knots_matrix, moments_vector)))
    return knots, quadrature_coefficients


def gauss_iqr(a, b, n):
    moments = [k[0] for k in [integrate.quad(weight_func, a, b, args=(j,)) for j in range(2*n)]]
    moments_matrix = np.zeros((n, n))
    for s in range(n):
        for j in range(n):
            moments_matrix[s][j] = moments[j+s]
    moments_vector = np.matrix([-moments[n+s] for s in range(n)]).transpose()
    knot_polynomial_coefficients = list(np.ravel(np.linalg.solve(moments_matrix, moments_vector)))
    knot_polynomial_coefficients.append(1)
    knot_polynomial_coefficients.reverse()
    knots = list(np.ravel(np.roots(knot_polynomial_coefficients)))
    knots_matrix = np.zeros((n, n))
    for s in range(n):
        for j in range(n):
            knots_matrix[s][j] = knots[j] ** s
    moments_vector = np.matrix(moments[:n]).transpose()
    quadrature_coefficients = list(np.ravel(np.linalg.solve(knots_matrix, moments_vector)))
    return knots, quadrature_coefficients


def interpolation_quadrature_rules(a, b, n, method=regular_iqr, composite=False):
    numerical_value = 0
    knots, quadrature_coefficients = method(a, b, n)
    for i in range(n):
        numerical_value += quadrature_coefficients[i] * function(knots[i])
    if not composite:
        sample_value = integrate.quad(wanted_function, a, b)[0]
        exact_error = abs(sample_value - numerical_value)
        methodical_error = methodical_error_estimation(a, b, knots)
        errors_difference = abs(exact_error - methodical_error)
        print("Quadrature coefficients: ", quadrature_coefficients,
              "\nExact quadrature coefficients", coefficients(a, b, knots))
        print("Resulting value: ", numerical_value, "\nExact value: ", sample_value, "\nMethodical Error: ",
              methodical_error, "\nExact error: ", exact_error, "\nDifference between methodical and exact errors: ",
              errors_difference)
    return numerical_value


def methodical_error_estimation(a, b, knots):
    x_ = sp.Symbol("x")
    f = -sp.Abs(sp.diff(function(0, diff=True), sp.Symbol("x"), len(knots)))
    f = sp.utilities.lambdify(x_, f)
    return (-optimize.minimize_scalar(f, bounds=(a, b), method='Bounded').fun / math.factorial(len(knots))) * \
        integrate.quad(estimator_func, a, b, args=(knots,))[0]


def runge(a, b, n, epsilon=10**-6, method=regular_iqr):
    s_h1 = 0
    s_h2 = 0
    k = 3
    l_ = 2
    m = aitken(a, b, n)
    h = (b - a) / k
    for i in range(k):
        s_h1 += interpolation_quadrature_rules(a + i * h, a + (i + 1) * h, n, method=method, composite=True)
        for j in range(l_):
            s_h2 += interpolation_quadrature_rules(a + (i * l_ + j) * h / l_, a + (i * l_ + j + 1) * h / l_, n,
                                                   method=method, composite=True)
    h_opt = h * ((epsilon * (1 - l_ ** (-m)) / abs(s_h2 - s_h1)) ** (1 / m))
    k_opt = math.floor((b - a) / h_opt)
    return h_opt, k_opt


def richardson(a, b, n, epsilon=10**-6, method=regular_iqr):
    sample_value = integrate.quad(wanted_function, a, b)[0]
    s_h1 = 0
    s_h2 = 0
    k = 3
    l_ = 2
    m = 2
    h_1 = (b - a) / k
    h_2 = h_1 / l_
    for i in range(k):
        s_h1 += interpolation_quadrature_rules(a + i * h_1, a + (i + 1) * h_1, n, method=method, composite=True)
        for j in range(l_):
            s_h2 += interpolation_quadrature_rules(a + (i * l_ + j) * h_2, a + (i * l_ + j + 1) * h_2, n,
                                                   method=method, composite=True)
    m = aitken(a, b, n)
    res = s_h2 + (s_h2 - s_h1)/(l_ ** m - 1)
    print("Richardson value: ", res, "\nRichardson error: ", sample_value - res)
    return res


def aitken(a, b, n, method=regular_iqr):
    s_h1 = 0
    s_h2 = 0
    s_h3 = 0
    k = 3
    l_ = 2
    h_1 = (b - a) / k
    h_2 = h_1 / l_
    h_3 = h_2 / l_
    for i in range(k):
        s_h1 += interpolation_quadrature_rules(a + i * h_1, a + (i + 1) * h_1, n, method=method, composite=True)
        for j in range(l_):
            s_h2 += interpolation_quadrature_rules(a + (i * l_ + j) * h_2, a + (i * l_ + j + 1) * h_2, n,
                                                   method=method, composite=True)
            for _ in range(l_):
                s_h3 += interpolation_quadrature_rules(a + (i * l_ ** 2 + j * l_ + _) * h_3,
                                                       a + (i * l_ ** 2 + j * l_ + _ + 1) * h_3, n,
                                                       method=method, composite=True)
    m = - (np.log(abs((s_h3 - s_h2)/(s_h2 - s_h1)))/np.log(l_))
    return m


def composite_quadrature_rules(a, b, n, method=regular_iqr, accuracy_rule=runge, epsilon=10**-6):
    sample_value = integrate.quad(wanted_function, a, b)[0]
    numerical_value = 0
    h_opt, k_opt = accuracy_rule(a, b, n, epsilon=epsilon, method=method)
    for j in range(k_opt):
        numerical_value += interpolation_quadrature_rules(a + j * h_opt, a + (j + 1) * h_opt, n,
                                                          method=method, composite=True)
    numerical_value += interpolation_quadrature_rules(a + k_opt * h_opt, b, n, method=method, composite=True)
    print("\nComposite resulting value: ", numerical_value, "\nDifference: ", abs(sample_value - numerical_value))


def main(a, b, n):
    start = time.time()
    interpolation_quadrature_rules(a, b, n, method=regular_iqr)
    composite_quadrature_rules(a, b, n, method=regular_iqr, accuracy_rule=runge, epsilon=epsilon_)
    print("Elapsed time: ", time.time() - start)


if __name__ == "__main__":
    main(a_, b_, n_)

import math
from scipy import integrate, optimize
import numpy as np

epsilon = 10 ** -6
a = 1.8
b = 2.9
alpha = 0
beta = 4/7


def function(x, opt=False):
    if opt:
        return -(4 * math.cos(2.5*x) * math.exp(4*x/7) + 2.5 * math.sin(5.5*x) * math.exp(-3*x/5) + 4.3 * x)
    return 4 * math.cos(2.5*x) * math.exp(4*x/7) + 2.5 * math.sin(5.5*x) * math.exp(-3*x/5) + 4.3 * x


def weight_func(x, j):
    return ((x - a)**-alpha) * ((b-x)**-beta) * (x**j)


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


def coefficients(knots):
    coefficients_ = [k[0] for k in [integrate.quad(coefficients_func, a, b, args=(knots, j))
                                    for j in range(len(knots))]]
    return coefficients_


def interpolation_quadrature_rules(n):
    numerical_value = 0
    knots = np.linspace(a, b, n)
    moments = [k[0] for k in [integrate.quad(weight_func, a, b, args=(j,)) for j in range(n)]]
    knots_matrix = np.zeros((n, n))
    for s in range(n):
        for j in range(n):
            knots_matrix[s][j] = knots[j] ** s
    moments_vector = np.matrix([moments[i] for i in range(n)]).transpose()
    quadrature_coefficients = np.linalg.solve(knots_matrix, moments_vector)
    quadrature_coefficients = list(np.ravel(quadrature_coefficients))
    print("Quadrature coefficients: ", quadrature_coefficients, "\nExact quadrature coefficients", coefficients(knots))
    for i in range(n):
        numerical_value += quadrature_coefficients[i] * function(knots[i])
    sample_value = integrate.quad(wanted_function, a, b)[0]
    exact_error = abs(sample_value - numerical_value)
    methodical_error = methodical_error_estimation(knots, n)
    errors_difference = abs(exact_error - methodical_error)
    print("Resulting value: ", numerical_value, "\nExact value: ", sample_value, "\nMethodical Error: ",
          methodical_error, "\nExact error: ", exact_error, "\nDifference between methodical and exact errors: ",
          errors_difference)


def methodical_error_estimation(knots, n):
    return (math.ceil(-optimize.minimize_scalar(function, bounds=(a, b), method='Bounded', args=(True,)).fun*10) /
            (10*math.factorial(n)) * integrate.quad(estimator_func, a, b, args=(knots,))[0])


def main(n=3):
    interpolation_quadrature_rules(n)


if __name__ == "__main__":
    main()

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


def integrity(a, b, func, n):
    integral = 0
    step = (b - a) / n
    for i in range(n):
        integral += func(a+(i+0.5)*step)
    print("Tupoe resulting value: ", integral * step)


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


def interpolation_quadrature_rules(a, b, n, composite=False):
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


def composite_quadrature_rule(a, b, n, epsilon=10**-6):
    k = 5
    h = (b - a) / k
    numerical_value = 0
    for i in range(k):
        numerical_value += interpolation_quadrature_rules(a + i * h, a + (i + 1) * h, n, composite=True)
    print("\nComposite resulting value: ", numerical_value)


def richardson(a, b, epsilon):
    steps = 0
    m = 0
    while True: # compare to epsilon
        steps += 1
        h = (b - a) / steps
        r = steps - 1
        step_matrix = np.zeros((r+1, r+1))
        for i in range(r+1):
            for j in range(r+1):
                step_matrix[i][j] = 0


def methodical_error_estimation(a, b, knots):
    x_ = sp.Symbol("x")
    f = -sp.Abs(sp.diff(function(0, diff=True), sp.Symbol("x"), len(knots)))
    f = sp.utilities.lambdify(x_, f)
    # print(optimize.minimize_scalar(f, bounds=(a, b), method='Bounded'))
    # print(optimize.minimize_scalar(function, bounds=(a, b), method='Bounded', args=(True,)))
    # print((-optimize.minimize_scalar(function, bounds=(a, b), method='Bounded', args=(True,)).fun /
    #       math.factorial(len(knots))) * integrate.quad(estimator_func, a, b, args=(knots,))[0])
    return (-optimize.minimize_scalar(f, bounds=(a, b), method='Bounded').fun / math.factorial(len(knots))) * \
        integrate.quad(estimator_func, a, b, args=(knots,))[0]


def main(a, b, n):
    start = time.time()
    integrity(a, b, function, 100_000_00)
    interpolation_quadrature_rules(a, b, n)
    composite_quadrature_rule(a, b, n)
    print("Elapsed time: ", time.time() - start)


if __name__ == "__main__":
    main(a_, b_, n_)

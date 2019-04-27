import time
import numpy as np
import sympy as sp
from scipy import integrate, optimize
import warnings
warnings.filterwarnings("ignore")

epsilon_ = 10 ** -6
a_ = 1.8
b_ = 2.9
n_ = 3
alpha = 0
beta = 4/7


def function(x, opt=False, diff=False):
    if opt:
        return -abs(4 * np.cos(2.5*x) * np.exp(4*x/7) + 2.5 * np.sin(5.5*x) * np.exp(-3*x/5) + 4.3 * x)
    elif diff:
        x_ = sp.Symbol("x")
        return 4 * sp.cos(2.5 * x_) * sp.exp(4 * x_ / 7) + 2.5 * sp.sin(5.5 * x_) * sp.exp(-3 * x_ / 5) + 4.3 * x_
    return 4 * np.cos(2.5*x) * np.exp(4*x/7) + 2.5 * np.sin(5.5*x) * np.exp(-3*x/5) + 4.3 * x


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


def estimator_func(x, knots, power=1):
    return abs(weight_func(x, 0) * knot_polynomial(x, knots)**power)


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
    moments_vector = np.array(moments).transpose()
    quadrature_coefficients = list(np.ravel(np.linalg.solve(knots_matrix, moments_vector)))
    return knots, quadrature_coefficients, "standard"


def stand_meth_err_estimation(a, b, knots):
    x_ = sp.Symbol("x")
    f = -sp.Abs(sp.diff(function(0, diff=True), sp.Symbol("x"), len(knots)))
    f = sp.utilities.lambdify(x_, f)
    return (-optimize.minimize_scalar(f, bounds=(a, b), method='Bounded').fun / np.math.factorial(len(knots))) * \
        integrate.quad(estimator_func, a, b, args=(knots,))[0]


def gauss_iqr(a, b, n):
    moments = [k[0] for k in [integrate.quad(weight_func, a, b, args=(j,)) for j in range(2*n)]]
    moments_matrix = np.zeros((n, n))
    for s in range(n):
        for j in range(n):
            moments_matrix[s][j] = moments[j+s]
    moments_vector = np.array([-moments[n+s] for s in range(n)]).transpose()
    knot_polynomial_coefficients = list(np.ravel(np.linalg.solve(moments_matrix, moments_vector)))
    knot_polynomial_coefficients.append(1)
    knot_polynomial_coefficients.reverse()
    knots = list(np.ravel(np.roots(knot_polynomial_coefficients)))
    knots_matrix = np.zeros((n, n))
    for s in range(n):
        for j in range(n):
            knots_matrix[s][j] = knots[j] ** s
    moments_vector = np.array(moments[:n]).transpose()
    quadrature_coefficients = list(np.ravel(np.linalg.solve(knots_matrix, moments_vector)))
    return knots, quadrature_coefficients, "gauss"


def gauss_meth_err_estimation(a, b, knots):
    x_ = sp.Symbol("x")
    f = -sp.Abs(sp.diff(function(0, diff=True), sp.Symbol("x"), 2*len(knots)))
    f = sp.utilities.lambdify(x_, f)
    return (-optimize.minimize_scalar(f, bounds=(a, b), method='Bounded').fun / np.math.factorial(2*len(knots))) * \
        integrate.quad(estimator_func, a, b, args=(knots, 2))[0]


def interpolation_quadrature_rules(a, b, n, method=regular_iqr, composite=False):
    numerical_value = 0
    estimation_methods = {"standard": stand_meth_err_estimation, "gauss": gauss_meth_err_estimation}
    knots, quadrature_coefficients, est = method(a, b, n)
    for i in range(n):
        numerical_value += quadrature_coefficients[i] * function(knots[i])
    if not composite:
        sample_value = integrate.quad(wanted_function, a, b, epsabs=1e-10)[0]
        exact_error = abs(sample_value - numerical_value)
        methodical_error = estimation_methods[est](a, b, knots)
        print("Quadrature coefficients: ", quadrature_coefficients,
              "\nExact quadrature coefficients: ", coefficients(a, b, knots))
        print("Resulting value: ", numerical_value, "\nExact value: ", sample_value, "\nMethodical Error: ",
              methodical_error, "\nExact error: ", exact_error)
    return numerical_value


def s_h_values(a, b, n, k, method=regular_iqr):
    h = (b - a) / k
    s_h = 0
    for i in range(k):
        s_h += interpolation_quadrature_rules(a + i * h, a + (i + 1) * h, n, method=method, composite=True)
    return s_h


def runge(a, b, n, k=2, epsilon=10**-6, method=regular_iqr, accuracy=True):
    l_ = 2
    m = n-1
    h = (b - a) / k
    s_h1, s_h2 = s_h_values(a, b, n, k, method=method), s_h_values(a, b, n, k * l_, method=method)
    if accuracy:
        error = abs((s_h2 - s_h1) / (1 - l_ ** - m))
        return error
    h_opt = 0.95 * h * ((epsilon * (1 - l_ ** (-m)) / abs(s_h2 - s_h1)) ** (1 / m))
    k_opt = np.math.ceil((b - a) / h_opt)
    h_opt = (b - a) / k_opt
    return h_opt, k_opt


def richardson(a, b, n, r=2, epsilon=10**-6, method=regular_iqr, accuracy=True):
    l_ = 2
    m = n-1
    h = (b - a) / r
    error = epsilon + 1
    if accuracy:
        h_matrix = np.zeros((r + 1, r + 1))
        s_h_vector = []
        for i in range(r + 1):
            for j in range(r + 1):
                if j == r:
                    h_matrix[i][j] = -1
                else:
                    h_matrix[i][j] = (h / l_ ** i) ** (m + j)
            s_h_vector.append(- s_h_values(a, b, n, r * l_ ** i, method=method))
        s_h_vector = np.array(s_h_vector).transpose()
        coefficients_ = np.linalg.solve(h_matrix, s_h_vector)
        h_matrix[0][-1] = 0
        error = abs(list(np.ravel(h_matrix[0]*coefficients_))[0])
        return error
    r = 1
    s_h_vector = [- s_h_values(a, b, n, r*l_**0, method=method), - s_h_values(a, b, n, r*l_**r, method=method)]
    coefficients_ = []
    while error > epsilon:
        r += 1
        h_matrix = np.zeros((r + 1, r + 1))
        s_h_vector = list(s_h_vector)
        for i in range(r+1):
            for j in range(r+1):
                if j == r:
                    h_matrix[i][j] = -1
                else:
                    h_matrix[i][j] = (h/l_**i)**(m+j)
        s_h_vector.append(- s_h_values(a, b, n, r*l_**r, method=method))
        s_h_vector = np.array(s_h_vector).transpose()
        coefficients_ = np.linalg.solve(h_matrix, s_h_vector)
        h_matrix[-1][-1] = 0
        error = abs(list(np.ravel(h_matrix[-1] * coefficients_))[0])
    return coefficients_[-1], error


def aitken(a, b, n, method=regular_iqr):
    k = 3
    l_ = 2
    s_h1, s_h2, s_h3 = s_h_values(a, b, n, k, method=method), s_h_values(a, b, n, k * l_, method=method), \
        s_h_values(a, b, n, k * l_**2, method=method)
    m = - (np.log(abs((s_h3 - s_h2)/(s_h2 - s_h1)))/np.log(l_))
    return m


def composite_quadrature_rules(a, b, n, epsilon=10**-6, method=regular_iqr, accuracy_rule=runge, k=0):
    sample_value = integrate.quad(wanted_function, a, b, epsabs=1e-10)[0]
    if k:
        estimated_error = accuracy_rule(a, b, n, k, epsilon=epsilon, method=method, accuracy=True)
        numerical_value = s_h_values(a, b, n, k, method=method)
    elif accuracy_rule == richardson:
        numerical_value, estimated_error = accuracy_rule(a, b, n, epsilon=epsilon, method=method, accuracy=False)
    else:
        h_opt, k_opt = accuracy_rule(a, b, n, epsilon=epsilon, method=method, accuracy=False)
        estimated_error = accuracy_rule(a, b, n, k_opt, epsilon=epsilon, method=method, accuracy=True)
        numerical_value = s_h_values(a, b, n, k_opt, method=method)
    print("\nComposite resulting value: ", numerical_value, "\nExact value: ", sample_value,
          "\nEstimated error: ", estimated_error, "\nExact error: ", abs(sample_value - numerical_value))


def main(a, b, n):
    start = time.monotonic()
    interpolation_quadrature_rules(a, b, n, method=regular_iqr, composite=False)
    composite_quadrature_rules(a, b, n, epsilon=epsilon_, method=regular_iqr, accuracy_rule=runge, k=0)
    print("\nElapsed time: ", time.monotonic() - start)


if __name__ == "__main__":
    main(a_, b_, n_)

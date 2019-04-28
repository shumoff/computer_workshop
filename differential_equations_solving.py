import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import integrate, optimize
from scipy.misc import derivative

c_2 = 1/8
A = 1/15
B = 1/20
epsilon = 1e-4
x_init = 0
y_init = np.array([B * np.pi, A * np.pi])
x_k = np.pi
accuracy_order = 2


def system(x, y, sym=False):
    if sym:
        return [A * sp.Symbol("y1"), -B * sp.Symbol("y0")]
    return np.array([A * y[1], -B * y[0]])


def second_order_runge_kutta(k):
    x = [x_init]
    y = [y_init]
    a_2_1 = c_2
    b_2 = 1 / (2*c_2)
    b_1 = 1 - b_2
    h = (x_k - x[0]) / k
    for i in range(k):
        k_1 = h * system(x[i], y[i])
        k_2 = h * system(x[i] + c_2*h, y[i] + a_2_1*k_1)
        y.append(y[i] + b_1*k_1 + b_2*k_2)
        x.append(x[i] + h)
    return y


def fourth_order_runge_kutta(k):
    x = [x_init]
    y = [y_init]
    h = (x_k - x[0]) / k
    for i in range(k):
        k_1 = h * system(x[i], y[i])
        k_2 = h * system(x[i] + 0.5*h, y[i] + 0.5*k_1)
        k_3 = h * system(x[i] + 0.5*h, y[i] + 0.5*k_2)
        k_4 = h * system(x[i] + h, y[i] + k_3)
        y.append(y[i] + (1/6)*(k_1 + 2*k_2 + 2*k_3 + k_4))
        x.append(x[i] + h)
    return y


def runge_error(method=second_order_runge_kutta):
    r = 1
    k = 2
    approx_1 = method(k)[-1]
    approx_2 = method(k * (2 ** r))[-1]
    error = np.linalg.norm((approx_2 - approx_1) / (2 ** accuracy_order - 1))
    while error > epsilon:
        r += 1
        approx_1 = approx_2
        approx_2 = method(k * (2 ** r))[-1]
        error = np.linalg.norm((approx_2 - approx_1) / (2 ** accuracy_order - 1))
    return approx_2, error


def optimal_step(k, method=second_order_runge_kutta):
    approx_1 = method(k)[-1]
    approx_2 = method(k * 2)[-1]
    h = (x_k - x_init) / k
    h_opt = (h / 2) * (((2 ** accuracy_order - 1) * epsilon)
                       / np.linalg.norm(approx_2 - approx_1)) ** (1 / accuracy_order)
    k_opt = int(np.ceil((x_k - x_init) / h_opt))
    h_opt = (x_k - x_init) / k_opt
    return h_opt, k_opt


def error_by_step(title, method=second_order_runge_kutta):
    h_opt, k_opt = optimal_step(k=2, method=method)
    x_nodes = [x_init + h_opt*i for i in range(k_opt+1)]
    y_nodes = method(k=k_opt)
    errors = []
    for i in range(len(x_nodes)):
        errors.append(np.linalg.norm(y_nodes[i] - integrate.solve_ivp(
            system, (x_init, x_k), y_init, dense_output=True, atol=1e-12, rtol=1e-12).sol.__call__(x_nodes[i])))
    plt.plot(x_nodes, errors)
    plt.title(title)
    plt.grid(True)
    plt.show()


def main():
    answer_2nd = runge_error(method=second_order_runge_kutta)[0]
    answer_4th = runge_error(method=fourth_order_runge_kutta)[0]
    exact_answer = integrate.solve_ivp(
        system, (x_init, x_k), y_init, dense_output=True, atol=1e-12, rtol=1e-12).sol.__call__(x_k)
    absolute_error_2nd = np.linalg.norm(exact_answer - answer_2nd)
    absolute_error_4th = np.linalg.norm(exact_answer - answer_4th)
    h_opt_2nd, k_opt_2nd = optimal_step(k=2, method=second_order_runge_kutta)
    h_opt_4th, k_opt_4th = optimal_step(k=2, method=fourth_order_runge_kutta)
    answer_opt_2nd = second_order_runge_kutta(k=k_opt_2nd)[-1]
    answer_opt_4th = fourth_order_runge_kutta(k=k_opt_4th)[-1]
    absolute_error_opt_2nd = np.linalg.norm(exact_answer - answer_opt_2nd)
    absolute_error_opt_4th = np.linalg.norm(exact_answer - answer_opt_4th)
    print("Exact value at point: ", exact_answer,
          "\n2nd order value at point: ", answer_2nd, "\n4th order value at point: ", answer_4th,
          "\n2nd order absolute error: ", absolute_error_2nd, "\n4th order absolute error: ", absolute_error_4th,
          "\n\n2nd order optimal step: ", h_opt_2nd, "\n4th order optimal step: ", h_opt_4th,
          "\n2nd order value at point with opt step: ", answer_opt_2nd,
          "\n4th order value at point with opt step: ", answer_opt_4th,
          "\n2nd order absolute error with opt step: ", absolute_error_opt_2nd,
          "\n4th order absolute error with opt step: ", absolute_error_opt_4th)
    error_by_step("2nd order absolute error by independent variable", method=second_order_runge_kutta)
    error_by_step("4th order absolute error by independent variable", method=fourth_order_runge_kutta)


if __name__ == "__main__":
    main()

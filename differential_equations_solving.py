import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import integrate

c_2 = 1 / 8
A = 1 / 15
B = 1 / 20
epsilon = 1e-4
delta = 1e-8
x_init = 0
y_init = np.array([B * np.pi, A * np.pi])
x_k = np.pi


def system(x, y, sym=False):
    if sym:
        return [A * sp.Symbol("y1"), -B * sp.Symbol("y0")]
    return np.array([A * y[1], -B * y[0]])


def second_order_runge_kutta(k, local_error=False, x_0=x_init, y_0=y_init):
    a_2_1 = c_2
    b_2 = 1 / (2 * c_2)
    b_1 = 1 - b_2
    x = []
    y = []
    y_nodes = []
    h = (x_k - x_init) / k
    if x_k - x_0 < h:
        h = x_k - x_0
    if local_error:
        k = 1
        counter = 2
    else:
        counter = 1
    for j in range(counter):
        x = [x_0]
        y = [y_0]
        h /= 2 ** j
        for i in range(k + j):
            k_1 = h * system(x[i], y[i])
            k_2 = h * system(x[i] + c_2 * h, y[i] + a_2_1 * k_1)
            y.append(y[i] + b_1 * k_1 + b_2 * k_2)
            x.append(x[i] + h)
        y_nodes.append(y[-1])
    if local_error:
        x_node = x[-1]
        return x_node, y_nodes
    return y


def fourth_order_runge_kutta(k, local_error=False, x_0=x_init, y_0=y_init):
    x = []
    y = []
    y_nodes = []
    h = (x_k - x_init) / k
    if x_k - x_0 < h:
        h = x_k - x_0
    if local_error:
        k = 1
        counter = 2
    else:
        counter = 1
    for j in range(counter):
        x = [x_0]
        y = [y_0]
        h /= 2 ** j
        for i in range(k + j):
            k_1 = h * system(x[i], y[i])
            k_2 = h * system(x[i] + 0.5 * h, y[i] + 0.5 * k_1)
            k_3 = h * system(x[i] + 0.5 * h, y[i] + 0.5 * k_2)
            k_4 = h * system(x[i] + h, y[i] + k_3)
            y.append(y[i] + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4))
            x.append(x[i] + h)
        y_nodes.append(y[-1])
    if local_error:
        x_node = x[-1]
        return x_node, y_nodes
    return y


def runge_error(approx_1, approx_2, method=second_order_runge_kutta):
    if method == second_order_runge_kutta:
        accuracy_order = 2
    else:
        accuracy_order = 4
    error = np.linalg.norm((approx_2 - approx_1) / (2 ** accuracy_order - 1))
    return error


def algorithm(method=second_order_runge_kutta):
    r = 1
    k = 2
    approx_1 = method(k)[-1]
    approx_2 = method(k * (2 ** r))[-1]
    error = runge_error(approx_1, approx_2, method=method)
    while error > epsilon:
        r += 1
        approx_1 = approx_2
        approx_2 = method(k * (2 ** r))[-1]
        error = runge_error(approx_1, approx_2, method=method)
    return approx_2, error


def optimal_step(k, method=second_order_runge_kutta):
    if method == second_order_runge_kutta:
        accuracy_order = 2
    else:
        accuracy_order = 4
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
    x_nodes = [x_init + h_opt * i for i in range(k_opt + 1)]
    y_nodes = method(k=k_opt)
    errors = []
    for i in range(len(x_nodes)):
        errors.append(np.linalg.norm(y_nodes[i] - integrate.solve_ivp(
            system, (x_init, x_k), y_init, dense_output=True, atol=epsilon * 10 ** -2, rtol=delta * 10 ** -2).
                                     sol.__call__(x_nodes[i])))
    plt.plot(x_nodes, errors)
    plt.title(title)
    plt.grid(True)
    plt.show()


def runge_local_error(y_nodes, method=second_order_runge_kutta):
    if method == second_order_runge_kutta:
        accuracy_order = 2
    else:
        accuracy_order = 4
    local_error = np.linalg.norm((y_nodes[1] - y_nodes[0]) / (1 - 2 ** - accuracy_order))
    return local_error


def auto_step_algorithm(title, method=second_order_runge_kutta):
    if method == second_order_runge_kutta:
        accuracy_order = 2
    else:
        accuracy_order = 4
    k = 2
    x_nodes = [x_init]
    y_nodes = [y_init]
    segment_length = x_k - x_init
    steps = [k]
    denied_steps = []
    denied_x = []
    counter = 0
    while x_k > x_nodes[-1]:
        counter += 1
        x_node, y_values = method(k, local_error=True, x_0=x_nodes[-1], y_0=y_nodes[-1])
        local_error = runge_local_error(y_values, method=method)
        if local_error > delta * 2 ** accuracy_order:
            denied_x.append(x_nodes[-1])
            denied_steps.append(k)
            k *= 2
        elif local_error > delta:
            x_nodes.append(x_node)
            y_nodes.append(y_values[0])
            steps.append(k)
            k *= 2
        elif local_error >= delta * (2 ** - (accuracy_order + 1)):
            x_nodes.append(x_node)
            y_nodes.append(y_values[0])
            steps.append(k)
        else:
            x_nodes.append(x_node)
            y_nodes.append(y_values[0])
            k = max((k / 2, min(steps)))
            steps.append(k)

    steps = [segment_length / counter for counter in steps]
    denied_steps = [segment_length / counter for counter in denied_steps]

    right_side_reference = counter * 3 * accuracy_order

    plt.plot(x_nodes, steps)
    plt.plot(denied_x, denied_steps, "rp")
    plt.title(f"{title} step length by independent variable")
    plt.grid(True)
    plt.show()

    errors = []
    for i in range(len(x_nodes)):
        errors.append(np.linalg.norm(y_nodes[i] - integrate.solve_ivp(
            system, (x_init, x_k), y_init, dense_output=True, atol=epsilon * 10 ** -2, rtol=delta * 10 ** -2).
                                     sol.__call__(x_nodes[i])))
    plt.plot(x_nodes, errors)
    plt.title(f"{title} auto-step absolute error by independent variable")
    plt.grid(True)
    plt.show()
    return y_nodes[-1], right_side_reference


def main():
    exact_answer = integrate.solve_ivp(
        system, (x_init, x_k), y_init, dense_output=True, atol=epsilon * 10 ** -2, rtol=delta * 10 ** -2).\
        sol.__call__(x_k)

    answer_2nd = algorithm(method=second_order_runge_kutta)[0]
    answer_4th = algorithm(method=fourth_order_runge_kutta)[0]
    absolute_error_2nd = np.linalg.norm(exact_answer - answer_2nd)
    absolute_error_4th = np.linalg.norm(exact_answer - answer_4th)

    h_opt_2nd, k_opt_2nd = optimal_step(k=2, method=second_order_runge_kutta)
    h_opt_4th, k_opt_4th = optimal_step(k=2, method=fourth_order_runge_kutta)
    answer_opt_2nd = second_order_runge_kutta(k=k_opt_2nd)[-1]
    answer_opt_4th = fourth_order_runge_kutta(k=k_opt_4th)[-1]
    absolute_error_opt_2nd = np.linalg.norm(exact_answer - answer_opt_2nd)
    absolute_error_opt_4th = np.linalg.norm(exact_answer - answer_opt_4th)

    error_by_step("2nd order absolute error by independent variable", method=second_order_runge_kutta)
    error_by_step("4th order absolute error by independent variable", method=fourth_order_runge_kutta)

    answer_auto_2nd, right_side_reference_2nd = auto_step_algorithm("2nd order", method=second_order_runge_kutta)
    answer_auto_4th, right_side_reference_4th = auto_step_algorithm("4th order", method=fourth_order_runge_kutta)
    absolute_error_auto_2nd = np.linalg.norm(exact_answer - answer_auto_2nd)
    absolute_error_auto_4th = np.linalg.norm(exact_answer - answer_auto_4th)

    print("Exact value at point: ", exact_answer,
          "\n2nd order value at point: ", answer_2nd, "\n4th order value at point: ", answer_4th,
          "\n2nd order absolute error: ", absolute_error_2nd, "\n4th order absolute error: ", absolute_error_4th,
          "\n\n2nd order optimal step: ", h_opt_2nd, "\n4th order optimal step: ", h_opt_4th,
          "\n2nd order value at point with opt step: ", answer_opt_2nd,
          "\n4th order value at point with opt step: ", answer_opt_4th,
          "\n2nd order absolute error with opt step: ", absolute_error_opt_2nd,
          "\n4th order absolute error with opt step: ", absolute_error_opt_4th,
          "\n\n2nd order value at point with auto step: ", answer_auto_2nd,
          "\n4th order value at point with auto-step: ", answer_auto_4th,
          "\n2nd order computations with auto-step: ", right_side_reference_2nd,
          "\n4th order computations with auto-step: ", right_side_reference_4th,
          "\n2nd order absolute error with auto-step: ", absolute_error_auto_2nd,
          "\n4th order absolute error with auto-step: ", absolute_error_auto_4th)


if __name__ == "__main__":
    main()

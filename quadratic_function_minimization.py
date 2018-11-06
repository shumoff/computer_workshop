from sympy import *
import time

N = 13
x, y, z = symbols('x y z')
func = 2*x**2 + (3 + 0.1*N) * y**2 + (4 + 0.1*N) * z**2 + x*y - y*z + x*z + x - 2*y + 3*z + N
A = Matrix([[4, 1, 1], [1, 2*(3 + 0.1*N), -1], [1, -1, 2*(4 + 0.1*N)]])
b = Matrix([1, -2, 3])
var = Matrix([x, y, z])
epsilon = 10**(-6)
point = Matrix([20, 20, 20])


def descent_gradient(point):
    start_time = time.time()
    i = 0
    grad_value = A * point + b
    while grad_value.norm() >= epsilon:
        a = A*var
        mu = - grad_value.norm()**2 / (grad_value.T * A * grad_value)[0]
        point = point + grad_value*mu
        i += 1
        if i % 100 == 0:
            print("x: ", point[0], "y: ", point[1], "z: ", point[2], "number of iterations: ", i)
        grad_value = A * point + b
    vec_norm = grad_value.norm()
    return [vec_norm, point, i, func.subs([(x, point[0]), (y, point[1]), (z, point[2])]), time.time() - start_time]


def coordinate_gradient(point):
    start_time = time.time()
    i = 0
    e = Matrix([1, 0, 0])
    grad_value = A * point + b
    while grad_value.norm()**2 >= epsilon:
        mu = - (e.T * grad_value)[0] / (e.T*A*e)[0]
        point = point + e*mu
        i += 1
        e[(i - 1) % 3] = 0
        e[i % 3] = 1
        if i % 100 == 0:
            print("x: ", point[0], "y: ", point[1], "z: ", point[2], "number of iterations: ", i)
        grad_value = A * point + b
    vec_norm = grad_value.norm()
    return [vec_norm, point, i, func.subs([(x, point[0]), (y, point[1]), (z, point[2])]), time.time() - start_time]


norma, point, i, func_value, elapsed_time = descent_gradient(point)
print("Point of minimum: ", "\n\tx: ", point[0], "\n\ty: ", point[1], "\n\tz: ", point[2],
      "\nFunction value: ", func_value, "\nNumber of iterations: ", i, "\nPrecision: ", norma,
      "\nElapsed time: ", elapsed_time)

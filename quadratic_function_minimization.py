from sympy import *
import time

N = 13
x, y, z = symbols('x y z')
func = 2*x**2 + (3 + 0.1*N) * y**2 + (4 + 0.1*N) * z**2 + x*y - y*z + x*z + x - 2*y + 3*z + N
A = Matrix([[2*2, 1, 1], [1, 2*(3 + 0.1*N), -1], [1, -1, 2*(4 + 0.1*N)]])
b = Matrix([1, -2, 3])
epsilon = 10**(-6)
point = Matrix([0, 0, 0])


def descent_gradient(point_):
    start_time = time.time()
    i_ = 0
    grad_value = A * point_ + b
    while grad_value.norm() >= epsilon:
        mu = - grad_value.norm()**2 / (grad_value.T * A * grad_value)[0]
        point_ += grad_value*mu
        i_ += 1
        if i_ % 100 == 0:
            print("x: ", point_[0], "y: ", point_[1], "z: ", point_[2], "number of iterations: ", i_)
        grad_value = A * point_ + b
    vec_norm = grad_value.norm()
    return [vec_norm, point_, i_, func.subs([(x, point_[0]), (y, point_[1]), (z, point_[2])]), time.time() - start_time]


def coordinate_gradient(point_):
    start_time = time.time()
    i_ = 0
    e = Matrix([1, 0, 0])
    grad_value = A * point_ + b
    while grad_value.norm()**2 >= epsilon:
        mu = - (e.T * grad_value)[0] / (e.T*A*e)[0]
        point_ += e*mu
        i_ += 1
        e[(i_ - 1) % 3] = 0
        e[i_ % 3] = 1
        if i_ % 100 == 0:
            print("x: ", point_[0], "y: ", point_[1], "z: ", point_[2], "number of iterations: ", i_)
        grad_value = A * point_ + b
    vec_norm = grad_value.norm()
    return [vec_norm, point_, i_, func.subs([(x, point_[0]), (y, point_[1]), (z, point_[2])]), time.time() - start_time]


norma, point, i, func_value, elapsed_time = descent_gradient(point)
print("Point of minimum: ", "\n\tx: ", point[0], "\n\ty: ", point[1], "\n\tz: ", point[2],
      "\nFunction value: ", func_value, "\nNumber of iterations: ", i, "\nPrecision: ", norma,
      "\nElapsed time: ", elapsed_time)

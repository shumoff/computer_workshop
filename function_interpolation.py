import numpy as np
import matplotlib.pyplot as plt
import math as m

a = 7
N = 50
def raz (x, y, k):
    z = 0
    for j in range(k):
        p = 1
        for i in range(k):
            if i == j:
                p = p * 1
            else:
                p = p * (x[j] - x[i])
        z = z + y[j] / p
    return z
def newton(x, y, t):
    z = y[0]
    for j in range(len(y) - 1):
        p = 1
        for i in range(j + 1):
            p = p * (t - x[i])
        z = z + raz(x, y, j + 2) * p
    return z

def f(x):
    return (3 * x - m.cos(x) - 1)


def chebishev(i):
    return (0.5 * ((a + a) * m.cos(m.pi * (2 * i + 1) / (2 * (N + 1)))))


def lagranzh(x, y, t):
    z = 0
    for j in range(len(y)):
        p1 = 1
        p2 = 1
        for i in range(len(x)):
            if i == j:
                p1 = p1 * 1
                p2 = p2 * 1
            else:
                p1 = p1 * (t - x[i])
                p2 = p2 * (x[j] - x[i])
        z = z + y[j] * (p1 / p2)
    return z


xnew = np.linspace(-a, a, 1000)
x = np.linspace(-a, a, N)
xCH = [chebishev(i) for i in range(N)]
y = [f(i) for i in x]
yCH = [f(i) for i in xCH]

yF = [f(i) for i in xnew]
yL = [lagranzh(x, y, i) for i in xnew]
yN = [newton(x, y, i) for i in xnew]
yCHL = [lagranzh(xCH, yCH, i) for i in xnew]
yCHN = [newton(xCH, yCH, i) for i in xnew]
c = [abs(yL[i] - yF[i]) for i in range(len(yL))]
print(max(c))
# plt.plot(xnew, yF, xnew, yN, xnew, yCHN)
plt.plot(xnew, yF, xnew, yL, xnew, yCHL)
plt.grid(True)
plt.show()
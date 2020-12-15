from math import sin, sqrt
import numpy as np

def F6(x, y):
    return 0.5 - (
        (((np.sin(np.sqrt(np.power(x, 2) + np.power(y, 2)))) ** 2) - (0.5))
        / (((1) + (0.001) * ((np.power(x, 2) + np.power(y, 2)))) ** 2)
    )

def F6_1(x, y):
    return 0.5 - (
        (((np.sin(np.sqrt(np.power(x, 2) + np.power(y, 2)))) ** 2) - (0.5))
        / (((1) + (0.001) * ((np.power(x, 2) + np.power(y, 2)))) ** 2)
    ) + 999.

def F6_2(x1, x2, x3, x4, x5):
    return F6(x1,x2) + F6(x2,x3) + F6(x3,x4) + F6(x4,x5) + F6(x5,x1)
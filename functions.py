from math import sin, sqrt
import numpy as np

def F6(x, y):
    return 0.5 - (
        (((np.sin(np.sqrt(np.power(x, 2) + np.power(y, 2)))) ** 2) - (0.5))
        / (((1) + (0.001) * ((np.power(x, 2) + np.power(y, 2)))) ** 2)
    )
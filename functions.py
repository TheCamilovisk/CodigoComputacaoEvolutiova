from math import sin, sqrt
import numpy as np

def F6(x1,x2):
    fitness = 0.5 - (
        (((sin(sqrt(x1 ** 2 + x2 ** 2))) ** 2) - (0.5))
        / (((1) + (0.001) * ((x1 ** 2 + x2 ** 2))) ** 2)
    )

    return fitness
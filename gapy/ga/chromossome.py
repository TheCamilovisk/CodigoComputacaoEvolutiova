"""
===========================================================================================
    ALGORITMO GENETICO BASICO - chromossome.py
===========================================================================================
Universidade Federal do Para - UFPA
Trabalho de Computacao Evolucionaria
Docente: Prof. Dr. Roberto Celio Limao de Oliveira
===========================================================================================
"""


import random as rd
from math import sin, sqrt
import numpy as np
from ..utils import utils


class Chromossome:
    """
    Classe para criar o cromossomo e gerar seu fitness

    Funciona de duas formas:
        1. ch = chromossome.Chromossome(nvar, lvar)
            Usado para gerar cromossomos com uma sequencia aleatoria

        2. ch = chromossome.Chromossome(nvar, lvar, sequence)
            Usado quando jah existe uma sequencia de cromossomos
            sequence => sequencia de cromossomos passada para a funcao
    """

    def __init__(self, nvar, lvar, ls, li, sequence=None):
        self.nvar = nvar
        self.lvar = lvar
        self.ls = ls
        self.li = li
        if sequence is None:
            self.sequence = utils.randomSequence(self.lvar)
        else:
            self.sequence = sequence
        self.sl = [0, 0]

        self.coords = utils.bin2real(
            self.nvar, self.lvar, self.sequence, self.ls, self.li
        )
        self.fitness = self.function()

    def function(self):
        # Funcao de avalicao do AG

        # print('X= {} \t Y= {} \t'.format(x,y))

        fitness = 0.5 - (
            (((sin(sqrt(self.coords[0] ** 2 + self.coords[1] ** 2))) ** 2) - (0.5))
            / (((1) + (0.001) * ((self.coords[0] ** 2 + self.coords[1] ** 2))) ** 2)
        )

        # print('F(X,Y): {} \t'.format(fitness))

        return fitness

        # return 0.5 + (sin((x**2)+(y**2))**2 - 0.5)/(1 + 0.001*sin((x**2)+(y**2))**2)

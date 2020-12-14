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

    def __init__(self, nvar, lvar, ls, li, function, sequence=None, representation=0):
        self.nvar = nvar
        self.lvar = lvar
        self.ls = ls
        self.li = li
        self.representation = representation
        if representation == 0: # Binary
            if sequence is None:
                self.sequence = utils.randomSequence(self.lvar)
            else:
                self.sequence = sequence
        else:                   # Real
            if sequence is None:
                self.sequence = utils.randomSequenceReal(self.lvar,self.ls,self.li)
            else:
                self.sequence = sequence
        self.sl = [0, 0]

        if representation == 0: # Binary
            self.coords = utils.bin2real(
                self.nvar, self.lvar, self.sequence, self.ls, self.li
            )
        else:                   # Real
            self.coords = self.sequence

        self.function = function
        self.fitness = self.function(*self.coords)

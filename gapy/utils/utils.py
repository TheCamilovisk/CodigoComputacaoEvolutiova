"""
===========================================================================================
    ALGORITMO GENETICO BASICO - aux.py
===========================================================================================
Universidade Federal do Para - UFPA
Trabalho de Computacao Evolucionaria
Docente: Prof. Dr. Roberto Celio Limao de Oliveira
===========================================================================================
"""

# Aqui encontram-se funcoes de auxilio ao Algoritmo Genetico

import random as rd
import numpy as np


def bin2real(nvar, lvar, sequence, ls, li):
    # Converte lista de binarios em lista de reais
    var = []
    varR = []
    sequence = list2str(sequence)

    b = 0
    for i in range(nvar):
        if i == 0:
            var = [sequence[: lvar[i]]]
        else:
            var += [sequence[b : b + lvar[i]]]
        b += lvar[i]

    # print('HERE: {}'.format(var))
    for i in range(nvar):
        varR += [
            float(li)
            + (float(ls) - float(li)) / ((2.0 ** lvar[i]) - 1) * float(int(var[i], 2))
        ]

    return varR


def randomSequence(lvar):
    # Gera a sequencia ramdomica de cromossomo
    x = np.ones(sum(lvar), int)

    return list(map(lambda i: x[i] * rd.randrange(0, 2), range(sum(lvar))))


def list2str(sequence):
    # Converte lista para string
    sequenceList = ""
    for i in sequence:
        sequenceList += str(i)

    return sequenceList


def varStr(lvar, sequence):
    # Retorna uma lista binaria de string

    varStr = []
    for i in range(len(lvar)):
        if i == 0:
            varStr = [sequence[0 : lvar[0]]]
        else:
            varStr = [sequence[lvar[i - 1] : lvar[i]]]

    return varStr

def normalization(x, min_pop, max_pop, maxVal_scale):
    return ((x-min_pop)/(max_pop-min_pop))/maxVal_scale
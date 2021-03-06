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

def randomSequenceReal(lvar, ls, li):
    return (np.random.rand(lvar,)*(ls-li)+li).tolist()

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

def normalization(x, pop_len, inc):
    return 1 + inc*x

def hamming_distance(seq1, seq2):
    assert len(seq1) == len(seq2)
    return sum(a1 != a2 for a1, a2 in zip(seq1, seq2))

def euclidian_distance(seq1, seq2):
    assert len(seq1) == len(seq2)
    squared_sums = [(a2 - a1) ** 2 for a1, a2 in zip(seq1, seq2)]
    return np.sqrt(np.sum(squared_sums))

def mdg_diversity(pop, repr):
    distances = []
    for i, crom1 in enumerate(pop[:-1]):
        for j, crom2 in enumerate(pop[i+1:]):
            if repr == 0:   # binary representation
                distances.append(hamming_distance(crom1.sequence, crom2.sequence))
            else:
                distances.append(euclidian_distance(crom1.coords, crom2.coords))
    
    len_pop = len(pop)
    diversity = np.sum(distances) * 2 / ((len_pop - 1) * len_pop)

    return diversity

def mda_diversity(pop):
    apt = np.array([c.fitness for c in pop])
    return apt.mean() / apt.max()

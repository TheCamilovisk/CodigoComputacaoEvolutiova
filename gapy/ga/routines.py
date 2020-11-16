"""
===========================================================================================
    ALGORITMO GENETICO BASICO - routines.py
===========================================================================================
Universidade Federal do Para - UFPA
Trabalho de Computacao Evolucionaria
Docente: Prof. Dr. Roberto Celio Limao de Oliveira
===========================================================================================
"""

import random as rd
import numpy as np
from ..utils import utils
import gapy.ga.chromossome as ch


def newGeneration(nvar, lvar, populationSize, ls, li):
    # Funcao paa gerar uma nova populacao randomica

    generation = []
    x = np.ones((sum(lvar)), int)
    i = 0

    while i < populationSize:
        generation += [ch.Chromossome(nvar, lvar, ls, li)]
        i += 1

    # Soma de todos os Fitness
    total = sum(map(lambda x: x.fitness, generation))

    # Laco que define a porcentagem de cada fitness
    for i in range(len(generation)):
        if i == 0:
            generation[i].sl = [0, generation[i].fitness / total]
        else:
            generation[i].sl = [
                generation[i - 1].sl[1],
                generation[i - 1].sl[1] + generation[i].fitness / total,
            ]

    return generation


def selection(generation):
    # Funcao de selecao de pais

    x = rd.random()
    y = rd.random()
    parents = []

    for i in range(len(generation)):
        if x > generation[i].sl[0] and x <= generation[i].sl[1]:
            parents.append(generation[i].sequence)
        else:
            pass

        if y > generation[i].sl[0] and y <= generation[i].sl[1]:
            parents.append(generation[i].sequence)
        else:
            pass

    return parents


def crossingOnePoint(nvar, lvar, tc, generation, ls, li):
    # Funcao de Crossover

    cGeneration = []
    i = 0

    while i < (len(generation) / 2):
        p = selection(generation)
        p1 = utils.list2str(p[0])
        p2 = utils.list2str(p[1])

        if rd.random() <= tc:
            point = rd.randint(1, sum(lvar) - 2)
            b1 = p1[:point] + p2[point:]
            b2 = p2[:point] + p1[point:]

            cGeneration += [
                ch.Chromossome(nvar, lvar, ls, li, list(b1)),
                ch.Chromossome(nvar, lvar, ls, li, list(b2)),
            ]

        else:
            cGeneration += [
                ch.Chromossome(nvar, lvar, ls, li, list(p1)),
                ch.Chromossome(nvar, lvar, ls, li, list(p2)),
            ]
        i += 1

    # Define a soma total de todos os Fitness
    total = sum(map(lambda x: x.fitness, cGeneration))

    for i in range(len(cGeneration)):
        if i == 0:
            cGeneration[i].sl = [0, cGeneration[i].fitness / total]
        else:
            cGeneration[i].sl = [
                cGeneration[i - 1].sl[1],
                cGeneration[i - 1].sl[1] + cGeneration[i].fitness / total,
            ]

    return cGeneration


def mutation(generation, tm):
    # Funcao de Mutacao

    for i in range(len(generation)):
        for j in range(len(generation[i].sequence)):
            x = rd.random()
            if x <= tm:
                if generation[i].sequence[j] == 0:
                    generation[i].sequence[j] = 1
                else:
                    generation[i].sequence[j] = 0
        generation[i].fitness = generation[i].function()

    return generation

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


def newGeneration(nvar, lvar, populationSize, ls, li, selectionMode, function, representation):
    # Funcao paa gerar uma nova populacao randomica

    generation = []
    # x = np.ones((sum(lvar)), int)
    i = 0
    
    while i < populationSize:
        generation += [ch.Chromossome(nvar, lvar, ls, li, function, None, representation)]
        i += 1

    if selectionMode == 0:
        # Soma de todos os Fitness
        total = sum(map(lambda x: x.fitness, generation))
        for i in range(len(generation)):
            if i == 0:
                generation[i].sl = [0, generation[i].fitness / total]
            else:
                generation[i].sl = [
                    generation[i - 1].sl[1],
                    generation[i - 1].sl[1] + generation[i].fitness / total,
                ]
    elif selectionMode != 0:
        generation.sort(key=lambda x: x.fitness, reverse=False)
        total = sum(map(lambda x: utils.normalization(x, len(generation), selectionMode), range(len(generation))))
        for i in range(len(generation)):
            if i == 0:
                generation[i].sl = [0, utils.normalization(i, len(generation), selectionMode) / total]
            else:
                generation[i].sl = [
                    generation[i - 1].sl[1],
                    generation[i - 1].sl[1] + utils.normalization(i, len(generation), selectionMode / total)
                ]

    return generation


def selection(generation, selectionMode=0):
    # Funcao de selecao de pais
    while True:
        if selectionMode == 0:
            x = rd.random()
            y = rd.random()
        else:
            x = rd.random()*generation[-1].sl[1]
            y = rd.random()*generation[-1].sl[1]
        parents = [0, 0]

        for i in range(len(generation)):
            if x > generation[i].sl[0] and x <= generation[i].sl[1]:
                parents[0] = i
            else:
                pass

            if y > generation[i].sl[0] and y <= generation[i].sl[1]:
                parents[1] = i
            else:
                pass
        if parents[0] != parents[1]:
            break

    return [generation[parents[0]].sequence, generation[parents[1]].sequence]


def elitism(generation, mode=1, gap=0):
    generation.sort(key=lambda x: x.fitness, reverse=True)

    if mode == 1:
        return generation[0]
    elif mode == 2:
        return generation[: int(gap * len(generation))]


def crossingOver(nvar, lvar, tc, generation, ls, li, function, selectionMode=0, crossingType=0):
    # Funcao de Crossover

    cGeneration = []
    i = 0

    while i < (len(generation) / 2):
        p = selection(generation, selectionMode)
        p1 = utils.list2str(p[0])
        p2 = utils.list2str(p[1])

        if rd.random() <= tc:
            if crossingType == 0:
                point = rd.randint(1, sum(lvar) - 2)
                b1 = p1[:point] + p2[point:]
                b2 = p2[:point] + p1[point:]
            elif crossingType == 1:
                point = [rd.randint(1, sum(lvar) - 2), rd.randint(1, sum(lvar) - 2)]
                if point[0] == point[1]:
                    while point[0] == point[1]:
                        point = [rd.randint(1, sum(lvar) - 2), rd.randint(1, sum(lvar) - 2)]
                point.sort()
                aux1 = p1[point[0]:point[1]]
                aux2 = p2[point[0]:point[1]]
                b1 = p1[:point[0]] + aux2 + p1[point[1]:]
                b2 = p2[:point[0]] + aux1 + p2[point[1]:]
            else:
                standard = [rd.randint(0,2) for i in range(sum(lvar))]
                b1 = [p1[i] if standard[i] == 1 else p2[i] for i in range(len(standard))]
                b2 = [p2[i] if standard[i] == 1 else p1[i] for i in range(len(standard))]

            cGeneration += [
                ch.Chromossome(nvar, lvar, ls, li, function, list(b1)),
                ch.Chromossome(nvar, lvar, ls, li, function, list(b2)),
            ]

        else:
            cGeneration += [
                ch.Chromossome(nvar, lvar, ls, li, function, list(p1)),
                ch.Chromossome(nvar, lvar, ls, li, function, list(p2)),
            ]
        i += 1

    if selectionMode == 0:
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
    else:
        cGeneration.sort(key=lambda x: x.fitness, reverse=False)
        total = sum(map(lambda x: utils.normalization(x, len(cGeneration), selectionMode), range(len(cGeneration))))

        for i in range(len(cGeneration)):
            if i == 0:
                cGeneration[i].sl = [0, utils.normalization(i, len(cGeneration), selectionMode) / total]
            else:
                cGeneration[i].sl = [
                    cGeneration[i - 1].sl[1],
                    cGeneration[i - 1].sl[1] + utils.normalization(i, len(cGeneration), selectionMode) / total
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
        generation[i].fitness = generation[i].function(*generation[i].coords)

    return generation

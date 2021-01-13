"""
===========================================================================================
    ALGORITMO GENETICO BASICO - geneticAlgorithms.py
===========================================================================================
Universidade Federal do Para - UFPA
Trabalho de Computacao Evolucionaria
Docente: Prof. Dr. Roberto Celio Limao de Oliveira
===========================================================================================
"""

# AQUI EH IMPLEMENTADA A CLASSE DO AG


import os

import matplotlib.pyplot as plt
import numpy as np
from functions import F6
from gapy.ga import routines
from gapy.utils import utils
from numpy.core.fromnumeric import argmax

objective = F6


class GeneticAlgorithms:
    def __init__(
        self,
        nvar,
        lvar,
        ngenerations,
        nruns,
        populationSize,
        nInitialPopulations,
        ls,
        li,
        function,
        elitism_mode=0,
        gap=0,
        tc=0.5,
        tm=8e-4,
        selectionMode=0,
        crossingType=0,
        representation=0
    ):

        self.nvar = nvar  # Number of variables
        self.lvar = lvar  # Variabels lengths
        self.ngenerations = ngenerations  # Number of generations
        self.nInitialPopulations = nInitialPopulations
        self.generations = []
        self.nruns = nruns  # Number of runs
        self.populationSize = populationSize  # Population size
        self.tc = tc  # Crossover rate
        self.tm = tm  # Mutation rate
        self.max = []  # Maximum
        self.min = []  # Minimum
        self.mean = []  # Mean
        self.std = []
        self.mdg = []
        self.mda = []
        self.ls = ls  # Upper bound
        self.li = li  # Lower bound
        self.var = 0
        self.elitism_mode = elitism_mode
        self.gap = gap
        self.selectionMode = selectionMode
        self.crossingType = crossingType
        self.function = function
        self.representation = representation

    def run(self):

        for epoch in range(self.nInitialPopulations):
            print("Epoch {}".format(epoch))
            n = 0

            initial = routines.newGeneration(
                        self.nvar, self.lvar, self.populationSize, self.ls, self.li, self.selectionMode, self.function, self.representation
                    )

            while n < self.nruns:
                # Creating generations
                self.generations.append(initial)
                # print("Run/Generation = {}/{}".format(n, 0))
                m = 1
                while m < self.ngenerations:
                    # Crossover and mutation routines
                    self.generations.append(routines.crossingOver(
                            self.nvar,
                            self.lvar,
                            self.tc,
                            self.generations[-1],
                            self.ls,
                            self.li,
                            self.function,
                            self.selectionMode,
                            self.crossingType,
                            self.representation,
                        ))
                    if self.representation == 0:
                        self.generations[-1] = routines.mutation(self.generations[-1], self.tm)
                    else:
                        self.generations[-1] = routines.mutation(self.generations[-1], self.tm, self.ls, self.li)
                    if self.elitism_mode == 0:
                        pass
                    elif self.elitism_mode == 1:
                        self.generations[-1][0] = routines.elitism(self.generations[-2])
                    elif self.elitism_mode == 2:
                        self.generations[-1][:int(self.gap*self.populationSize)] = routines.elitism(self.generations[-2], self.elitism_mode, self.gap)
                    # print("Run/Generation = {}/{}".format(n, m))
                    m += 1
                n += 1

            for i in range(0, len(self.generations), self.ngenerations):
                maximum = []
                minimum = []
                mean = []
                std = []

                mdg = []
                mda = []
                for j in range(i, i + self.ngenerations):
                    maximum += [float(max(map(lambda x: x.fitness, self.generations[j])))]
                    minimum += [float(min(map(lambda x: x.fitness, self.generations[j])))]
                    mean += [
                        float(
                            sum(
                                map(
                                    lambda x: x.fitness / self.populationSize,
                                    self.generations[j],
                                )
                            )
                        )
                    ]
                    std += [
                        float(
                            sum(
                                map(
                                    lambda x: (x.fitness - mean[-1]) ** 2
                                    / self.populationSize,
                                    self.generations[j],
                                )
                            )
                        )
                    ]
                    mdg += [utils.mdg_diversity(self.generations[j], self.representation)]
                    mda += [utils.mda_diversity(self.generations[j])]

                self.max += [maximum]
                self.min += [minimum]
                self.mean += [mean]
                self.std += [std]
                self.mdg += [mdg]
                self.mda += [mda]
                maximum = []
                minimum = []
                mean = []

    def plotting(self, output_folder, prefix):
        keys = range(self.ngenerations)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        mean_mdg = np.array(self.mdg).mean(0)
        std_mdg = np.array(self.mdg).std(0)
        ax.plot(mean_mdg, "k-", color="green")
        ax.fill_between(keys, mean_mdg - std_mdg, mean_mdg + std_mdg, facecolor="blue", alpha=0.5)
        ax.set_title("Diversidade genotípica entre as {} populações iniciais".format(self.nInitialPopulations))
        ax.set_xlabel("Gerações")
        ax.set_ylabel("Diversidade")
        ax.grid()
        fig_name = os.path.join(output_folder, f"{prefix}_mdg_50_geracoes.png")
        fig.savefig(fig_name, bbox_inches="tight")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        mean_mda = np.array(self.mda).mean(0)
        std_mda = np.array(self.mda).std(0)
        ax.plot(mean_mda, "k-", color="green")
        ax.fill_between(keys, mean_mda - std_mda, mean_mda + std_mda, facecolor="blue", alpha=0.5)
        ax.set_title("Diversidade fenotípica entre as {} populações iniciais".format(self.nInitialPopulations))
        ax.set_xlabel("Gerações")
        ax.set_ylabel("Diversidade")
        ax.grid()
        fig_name = os.path.join(output_folder, f"{prefix}_mda_50_geracoes.png")
        fig.savefig(fig_name, bbox_inches="tight")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        mean = np.array(self.mean).mean(0)
        max_val = np.array(self.max).mean(0)
        std = np.array(self.std).mean(0)
        ax.plot(mean, "k-", color="green")
        ax.plot(max_val, "k-", color="red")
        ax.fill_between(keys, mean - std, mean + std, facecolor="blue", alpha=0.5)
        ax.fill_between(keys, max_val - np.array(self.max).std(0), max_val + np.array(self.max).std(0), facecolor="blue", alpha=0.5)
        ax.set_title("Performance Média entre as {} populações iniciais".format(self.nInitialPopulations))
        ax.set_xlabel("Gerações")
        ax.set_ylabel("Aptidão")
        ax.grid()
        fig_name = os.path.join(output_folder, f"{prefix}_desempenho_50_geracoes.png")
        fig.savefig(fig_name, bbox_inches="tight")

        i = argmax([max(x) for x in self.max]) #cGeneration.sorted(key=lambda x: x.fitness, reverse=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        mean = np.array(self.mean[i])
        std = np.array(self.std[i])
        ax2.plot(mean, "k-", color="green")
        ax2.fill_between(keys, mean - std, mean + std, facecolor="blue", alpha=0.5)
        ax2.set_title("Performance Média")
        ax2.set_xlabel("Gerações")
        ax2.set_ylabel("Aptidão")
        ax2.grid()

        ax1.plot(keys, self.max[i], "k-", label="Maximum", color="blue")
        ax1.plot(keys, self.min[i], "k-", label="Minimum", color="green")
        ax1.plot(keys, self.mean[i], "k-", label="Mean", color="red")
        ax1.set_title("Performance (Max = {})".format(max(self.max[i])))
        ax1.set_xlabel("Gerações")
        ax1.set_ylabel("Aptidão")
        ax1.legend()
        ax1.grid()
        fig_name = os.path.join(output_folder, f"{prefix}_desempenho_1_geracao.png")
        fig.savefig(fig_name, bbox_inches="tight")

        if self.nvar <= 2:

            x = np.arange(-100, 100, 1)
            xx, yy = np.meshgrid(x, x)
            z = objective(xx, yy)

            gCoords = np.array([[c.coords for c in gen] for gen in self.generations])

            fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = plt.subplots(2, 3, figsize=(10, 6))
            fig.suptitle("AG Evolution")

            ax11.contourf(xx, yy, z)
            gen_i = 0
            gen = gCoords[gen_i]
            fitness = objective(gen[:, 0], gen[:, 1])
            max_x, max_y = gen[fitness.argmax()]
            min_x, min_y = gen[fitness.argmin()]
            ax11.scatter(gen[:, 0], gen[:, 1], c="black")
            ax11.scatter(max_x, max_y, c="blue")
            ax11.scatter(min_x, min_y, c="red")
            ax11.set(xlim=(-100, 99))
            ax11.set(ylim=(-100, 99))
            ax11.set(title=f"Generation {gen_i + 1}")

            ax12.contourf(xx, yy, z)
            gen_i = 9
            gen = gCoords[gen_i]
            fitness = objective(gen[:, 0], gen[:, 1])
            max_x, max_y = gen[fitness.argmax()]
            min_x, min_y = gen[fitness.argmin()]
            ax12.scatter(gen[:, 0], gen[:, 1], c="black")
            ax12.scatter(max_x, max_y, c="blue")
            ax12.scatter(min_x, min_y, c="red")
            ax12.set(xlim=(-100, 99))
            ax12.set(ylim=(-100, 99))
            ax12.set(title=f"Generation {gen_i + 1}")

            ax13.contourf(xx, yy, z)
            gen_i = 19
            gen = gCoords[gen_i]
            fitness = objective(gen[:, 0], gen[:, 1])
            max_x, max_y = gen[fitness.argmax()]
            min_x, min_y = gen[fitness.argmin()]
            ax13.scatter(gen[:, 0], gen[:, 1], c="black")
            ax13.scatter(max_x, max_y, c="blue")
            ax13.scatter(min_x, min_y, c="red")
            ax13.set(xlim=(-100, 99))
            ax13.set(ylim=(-100, 99))
            ax13.set(title=f"Generation {gen_i + 1}")

            ax21.contourf(xx, yy, z)
            gen_i = 29
            gen = gCoords[gen_i]
            fitness = objective(gen[:, 0], gen[:, 1])
            max_x, max_y = gen[fitness.argmax()]
            min_x, min_y = gen[fitness.argmin()]
            ax21.scatter(gen[:, 0], gen[:, 1], c="black")
            ax21.scatter(max_x, max_y, c="blue")
            ax21.scatter(min_x, min_y, c="red")
            ax21.set(xlim=(-100, 99))
            ax21.set(ylim=(-100, 99))
            ax21.set(title=f"Generation {gen_i + 1}")

            ax22.contourf(xx, yy, z)
            gen_i = 39
            gen = gCoords[gen_i]
            fitness = objective(gen[:, 0], gen[:, 1])
            max_x, max_y = gen[fitness.argmax()]
            min_x, min_y = gen[fitness.argmin()]
            ax22.scatter(gen[:, 0], gen[:, 1], c="black")
            ax22.scatter(max_x, max_y, c="blue")
            ax22.scatter(min_x, min_y, c="red")
            ax22.set(xlim=(-100, 99))
            ax22.set(ylim=(-100, 99))
            ax22.set(title=f"Generation {gen_i + 1}")

            ax23.contourf(xx, yy, z)
            gen_i = 49
            gen = gCoords[gen_i]
            fitness = objective(gen[:, 0], gen[:, 1])
            max_x, max_y = gen[fitness.argmax()]
            min_x, min_y = gen[fitness.argmin()]
            ax23.scatter(gen[:, 0], gen[:, 1], c="black")
            ax23.scatter(max_x, max_y, c="blue")
            ax23.scatter(min_x, min_y, c="red")
            ax23.set(xlim=(-100, 99))
            ax23.set(ylim=(-100, 99))
            ax23.set(title=f"Generation {gen_i + 1}")

            fig_name = os.path.join(output_folder, f"{prefix}_evolucao.png")
            fig.savefig(fig_name, bbox_inches="tight")

        plt.show()

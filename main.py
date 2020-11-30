"""
===========================================================================================
    BASIC GENETIC ALGORITHM - main.py
===========================================================================================
Federal University of Pará- UFPA
Professor: Prof. Dr. Roberto Celio Limao de Oliveira
Student: Luan Assis Gonçalves
===========================================================================================
"""

from os import read
from numpy.lib.shape_base import split
from gapy.ga import geneticAlgorithms
from gapy.ga import routines
import numpy as np
import argparse


def calcNumberOfBits(precisions, lowerBound, upperBound):
    lvar = []
    for i in range(len(precisions)):
        lvar.append(
            int(
                np.ceil(
                    np.log2((upperBound - lowerBound) * np.power(10, precisions[i]))
                )
            )
        )
    return lvar

def readCFG(cfg_file):
    cfg_file = open(cfg_file, "r")
    cfg_file = cfg_file.read().split("\n")
    cfg = []
    for i in range(len(cfg_file)):
        if cfg_file[i].split("=")[-1].find(".") != -1:
            cfg.append(float(cfg_file[i].split("=")[-1]))
        elif cfg_file[i].split("=")[-1].find(",") != -1:
            cfg.append(list(map(int,cfg_file[i].split("=")[-1].split(","))))
        elif cfg_file[i].split("=")[-1].find(".") == -1:
            cfg.append(int(cfg_file[i].split("=")[-1]))
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--nvar", help="Number of variables", type=int, required=True)
    # parser.add_argument(
    #     "--precisions", help="Precision of the variables.", type=str, required=True
    # )
    # parser.add_argument(
    #     "--ngenerations",
    #     help="Number of generations",
    #     type=int,
    #     required=False,
    #     default=1000,
    # )
    # parser.add_argument(
    #     "--nruns", help="Number of runs", type=int, required=False, default=5
    # )
    # parser.add_argument(
    #     "--populationSize",
    #     help="Population size",
    #     type=int,
    #     required=False,
    #     default=100,
    # )
    # parser.add_argument(
    #     "--nInitialPopulations",
    #     help="Number of initial populations",
    #     type=int,
    #     required=False,
    #     default=10,
    # )
    # parser.add_argument(
    #     "--upperBound", help="Upper bound", type=int, required=False, default=100
    # )
    # parser.add_argument(
    #     "--lowerBound", help="Lower bound", type=int, required=False, default=-100
    # )
    # parser.add_argument(
    #     "--tc", help="Crossover rate", type=float, required=False, default=75e-2
    # )
    # parser.add_argument(
    #     "--tm", help="Crossover rate", type=float, required=False, default=1e-2
    # )
    # parser.add_argument(
    #     "--em", help="Elitism mode", type=int, required=False, default=0
    # )
    # parser.add_argument("--gap", help="GAP", type=float, required=False, default=0)
    # parser.add_argument("--selectionMode", help="Selection mode: 0) fintness x) liner normalization maximum", type=int, required=False, default=0)
    # parser.add_argument("--crossingType", help="Crossing over mode: 0) One point 1) Two points and 2) Uniform", type=int, required=False, default=0)
    parser.add_argument("--cfg_file", help="Path to config file", type=str, required=True)
    args = parser.parse_args()

    nvar,lowerBound,upperBound,precisions,ngenerations,nruns,populationSize,nInitialPopulations,em,gap,tc,tm,selectionMode,crossingType=readCFG(args.cfg_file)

    # precisions = [int(x) for x in eval(precisions)]
    ag = geneticAlgorithms.GeneticAlgorithms(
        nvar,
        calcNumberOfBits(precisions, lowerBound, upperBound),
        ngenerations,
        nruns,
        populationSize,
        nInitialPopulations,
        lowerBound,
        upperBound,
        em,
        gap,
        tc,
        tm,
        selectionMode,
        crossingType,
    )
    ag.run()
    ag.plotting()

"""
===========================================================================================
    BASIC GENETIC ALGORITHM - main.py
===========================================================================================
Federal University of Pará- UFPA
Professor: Prof. Dr. Roberto Celio Limao de Oliveira
Student: Luan Assis Gonçalves
===========================================================================================
"""

import argparse

import numpy as np

from functions import F6
from gapy.ga import geneticAlgorithms


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
        if i == len(cfg_file)-2:
            cfg.append(float(cfg_file[i].split("=")[-1]))
        elif cfg_file[i].split("=")[-1].find(".") != -1:
            cfg.append(float(cfg_file[i].split("=")[-1]))
        elif cfg_file[i].split("=")[-1].find(",") != -1:
            cfg.append(list(map(int,cfg_file[i].split("=")[-1].split(","))))
        elif cfg_file[i].split("=")[-1].find(".") == -1:
            cfg.append(int(cfg_file[i].split("=")[-1]))
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", help="Path to config file", type=str, required=True)
    args = parser.parse_args()

    nvar,lowerBound,upperBound,precisions,ngenerations,nruns,populationSize,nInitialPopulations,em,gap,tc,tm,selectionMode,crossingType,representation=readCFG(args.cfg_file)

    # precisions = [int(x) for x in eval(precisions)]
    ag = geneticAlgorithms.GeneticAlgorithms(
        nvar,
        nvar if representation == 1 else calcNumberOfBits(precisions, lowerBound, upperBound),
        ngenerations,
        nruns,
        populationSize,
        nInitialPopulations,
        lowerBound,
        upperBound,
        F6,
        em,
        gap,
        tc,
        tm,
        selectionMode,
        crossingType,
        representation,
    )
    ag.run()
    ag.plotting()

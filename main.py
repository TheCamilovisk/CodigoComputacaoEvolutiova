"""
===========================================================================================
    BASIC GENETIC ALGORITHM - main.py
===========================================================================================
Federal University of Pará- UFPA
Professor: Prof. Dr. Roberto Celio Limao de Oliveira
Student: Luan Assis Gonçalves
===========================================================================================
"""

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvar", help="Number of variables", type=int, required=True)
    parser.add_argument(
        "--precisions", help="Precision of the variables.", type=str, required=True
    )
    parser.add_argument(
        "--ngenerations",
        help="Number of generations",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--nruns", help="Number of runs", type=int, required=False, default=5
    )
    parser.add_argument(
        "--populationSize",
        help="Population size",
        type=int,
        required=False,
        default=100,
    )
    parser.add_argument(
        "--nInitialPopulations",
        help="Number of initial populations",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "--upperBound", help="Upper bound", type=int, required=False, default=100
    )
    parser.add_argument(
        "--lowerBound", help="Lower bound", type=int, required=False, default=-100
    )
    parser.add_argument(
        "--tc", help="Crossover rate", type=float, required=False, default=75e-2
    )
    parser.add_argument(
        "--tm", help="Crossover rate", type=float, required=False, default=1e-2
    )
    parser.add_argument(
        "--em", help="Elitism mode", type=int, required=False, default=0
    )
    parser.add_argument("--gap", help="GAP", type=float, required=False, default=0)
    args = parser.parse_args()
    precisions = [int(x) for x in eval(args.precisions)]
    ag = geneticAlgorithms.GeneticAlgorithms(
        args.nvar,
        calcNumberOfBits(precisions, args.lowerBound, args.upperBound),
        args.ngenerations,
        args.nruns,
        args.populationSize,
        args.nInitialPopulations,
        args.lowerBound,
        args.upperBound,
        args.em,
        args.gap,
        args.tc,
        args.tm,
    )
    ag.run()
    ag.plotting()

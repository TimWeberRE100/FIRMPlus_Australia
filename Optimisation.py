# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import datetime as dt
import pygmo as pg
from numba import jit, float64
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', default=1000, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=100, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')
parser.add_argument('-s', default=11, type=int, required=False, help='11, 12, 13, ...')
parser.add_argument('-n', default='Super1', type=str, required=False, help='node=Super1')
parser.add_argument('-w', default=1, type=int, required=False, help='Number of islands in differential evolution (i.e. workers)')
args = parser.parse_args()

scenario = args.s
node = args.n

from Input import *
from Simulation import Reliability
from Network import Transmission

@jit(nopython=True)

def F(x):
    """This is the objective function."""

    S = Solution(x)

    Deficit = Reliability(S, flexible=np.zeros(intervals, dtype=np.float64)) # Sj-EDE(t, j), MW
    Flexible = Deficit.sum() * resolution / years / efficiency # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
    PenHydro = max(0, Hydro - 20 * 1000000) # TWh p.a. to MWh p.a.

    TDC = Transmission(S) if 'Super' in node else np.zeros((intervals, len(DCloss)), dtype=np.float64)  # TDC: TDC(t, k), MW
    TDC_abs = np.abs(TDC)

    Deficit = Reliability(S, flexible=np.ones(intervals, dtype=np.float64)*CPeak.sum()*1000) # Sj-EDE(t, j), GW to MW
    Deficit_sum = Deficit.sum() * resolution
    PenDeficit = max(0, Deficit_sum) # MWh

    CDC = np.zeros(len(DCloss), dtype=np.float64)
    for i in range(0,intervals):
        for j in range(0,len(DCloss)):
            if TDC_abs[i][j] > CDC[j]:
                CDC[j] = TDC_abs[i][j]
    CDC = CDC * 0.001 # CDC(k), MW to GW

    cost = factor *  np.concatenate((np.array([S.CPV.sum(), S.CWind.sum(), S.CPHP.sum(), S.CPHS]), CDC, np.array([S.CPV.sum(), S.CWind.sum(), Hydro * 0.000001, -1.0, -1.0])))
    cost = cost.sum()

    loss = TDC_abs.sum(axis=0) * DCloss
    loss = loss.sum() * 0.000000001 * resolution / years # PWh p.a.
    LCOE = cost / abs(energy - loss)

    Func = LCOE + PenDeficit + PenHydro

    return Func

if __name__=='__main__':
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    lb = [0.]  * pzones + [0.]   * wzones + contingency   + [0.]
    ub = [50.] * pzones + [50.]  * wzones + [50.] * nodes + [5000.]

    class EnergyOptimizationProblem:
        def __init__(self, lb, ub):
            self.lb = lb
            self.ub = ub
        
        def fitness(self, x):
            return EnergyOptimizationProblem._fitness(x)
        
        @jit(float64[:](float64[:]), nopython=True)
        def _fitness(x):
            retval = np.zeros((1,))
            retval[0] = F(x)
            # Your objective function F(x) goes here, return a tuple with one element
            return retval

        def get_bounds(self):
            # Return the bounds as tuples of (lb, ub)
            return (self.lb, self.ub)

        def get_nobj(self):
            # Return the number of objectives
            return 1

    prob = pg.problem(EnergyOptimizationProblem(lb, ub))

    algo = pg.algorithm(pg.de(gen=args.i, F=args.m, CR=args.r))
    algo.set_verbosity(1)  # Change verbosity level to control the amount of logging

    if args.w > 1:
        # Number of islands in the archipelago
        n_islands = 8  # You can adjust this based on your system's capabilities

        # Create an archipelago with the specified number of islands
        archi = pg.archipelago(n=n_islands, algo=algo, prob=prob, pop_size=args.p)

        # Evolve the archipelago in parallel
        archi.evolve()

        # Wait for the evolution to complete
        archi.wait()

        # Collect the results
        # You can inspect each island's best solution or aggregate results as needed
        for i, isl in enumerate(archi):
            print(f"Island {i}: Best Fitness = {isl.get_population().champion_x} {isl.get_population().champion_f}")

    else:
        pop = pg.population(prob, size=args.p)
        pop = algo.evolve(pop)

        best_solution = pop.champion_x
        best_solution_fitness = pop.champion_f[0]  # Assuming a single-objective problem

        # Print the best solution and its objective function value
        print("Best solution:", best_solution)
        print("Value of the objective function:", best_solution_fitness)

    """ with open('Results/Optimisation_resultx{}{}.csv'.format(args.n, args.e), 'a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(best_solution) """

    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

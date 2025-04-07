import csv
from datetime import datetime as dt

import numpy as np
from numba import njit, prange
from scipy.optimize import differential_evolution

from firm.Input import (
    Evaluate, 
    Solution, 
    cost_model, 
    x0, 
    lb, 
    ub, 
    scenario, 
    pzones, 
    wzones, 
    nodes, 
    nhvi,
    )
from firm.Parameters import Parameters, DE_Hyperparameters
from firm.Fileprinter import Fileprinter

def ObjectiveWrapper(xs, fileprinter, cost_model, y, p):
    result = ObjectiveParallel(xs.T, cost_model, y, p)
    fileprinter(np.vstack((np.atleast_2d(result), xs)).T) 
    return result
 
@njit(parallel=True)
def ObjectiveParallel(xs, cost_model, y, p):
    result = np.empty(len(xs), dtype=np.float64)
    for i in prange(len(xs)):
        result[i] = Objective(xs[i], cost_model, y, p)
    return result
 
@njit
def Objective(x, cost_model, y, p):
    """This is the objective function"""
    S = Solution(x, y, p)
    Evaluate(S, cost_model)
    return S.LCOE+S.Penalties

class CallbackClass:
    def __init__(
            self, 
            display: int = 50, 
            stagnation: int = 100, 
            stag_rate: float = 1e-6
            ):
        """
        This object is called after each iteration.
        display    - how often (# iterations) to print intermediate results to console
        stagnation - # of iterations with no improvement to best objective after which to terminate
        """
        self.it = 0 
        self.display = display
        self.stagnation = stagnation
        self.stag_counter = 0
        self.stag_rate = stag_rate
        self.start = dt.now()
        self.elite = np.inf
    
    def __call__(self, intermediate_result):
        if self.display > 0:
            if self.it % self.display == 0:
                print(f'Iteration: {self.it}. Time taken: {dt.now()-self.start}. Best value: {intermediate_result.fun}')
        if self.stag_rate > 0 and self.stagnation > 1:
            if self.elite - intermediate_result.fun < self.stag_rate:
                self.stag_counter+=1
            else: 
                self.elite = intermediate_result.fun
                self.stag_counter=0
            if self.stag_counter == self.stagnation:
                if self.display > 0:
                    print(f'Iteration: {self.it}. Time taken: {dt.now()-self.start}. Best value: {intermediate_result.fun}')
                return True
        self.it+=1
        return False

def Optimise(parameters, hyperparameters):
    fileprinter = Fileprinter(
        f"../Results/History{scenario}.csv", 
        hyperparameters.f, 
        header = ["Obj"] + 
                 [f"PV{n}" for n in range(pzones)] + 
                 [f"Wind{n}" for n in range(wzones)] +
                 [f"PHP{n}" for n in range(nodes)] + 
                 [f"PHE{n}" for n in range(nodes)] + 
                 [f"HVI{n}" for n in range(nhvi)],
        resume=False,
        )
    
    starttime = dt.now()
    print("Optimisation starts at", starttime)
    result = differential_evolution(
        func=ObjectiveWrapper,
        args=(
            fileprinter,
            cost_model, 
            parameters.y, 
            parameters.p,
            ),
        bounds=list(zip(lb, ub)),
        tol=0,
        maxiter=hyperparameters.i,
        popsize=hyperparameters.p,
        mutation=hyperparameters.m,
        recombination=hyperparameters.r,
        disp=False,
        callback=CallbackClass(
            hyperparameters.v, 
            *hyperparameters.s,
            ),
        polish=False,
        updating="deferred",
        x0=x0,
        vectorized=True
    )

    endtime = dt.now()
    timetaken = endtime - starttime
    print("Optimisation took", timetaken)

    with open(f"../Results/Optimisation_resultx{scenario}.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.x)

    return result, timetaken


if __name__ == "__main__":
    parameters = Parameters(y=1, p=False)
    hyperparameters = DE_Hyperparameters(
        i = 10, 
        p = 10, 
        m = (0.5, 1.0), 
        r = 0.4, 
        v = 5,
        s = (10, 1),
        f = 1,
        )
    
    
    print(Objective(x0, cost_model, parameters.y, parameters.p))
    
    result, time = Optimise(parameters, hyperparameters)




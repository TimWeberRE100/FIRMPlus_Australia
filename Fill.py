import numpy as np 
from numba import njit
import datetime as dt

from Simulation import Simulate
from Resimulation import Resimulate

@njit 
def Fill(solution):
    flexible = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)
    Resimulate(solution, flexible=flexible)
    
    fill = np.zeros(solution.nodes, np.float64)
    for t in range(solution.intervals-1, -1, -1):
        if (solution.Deficit[t] > 0).any():
            flexible[t] = np.minimum(solution.Deficit[t], solution.CPeak)
            fill += (solution.Deficit[t]-flexible[t])/solution.efficiency

        if fill.sum() > 1e-6:
            # simplified charging model
            fill = np.minimum(fill, (solution.CPHS - solution.Storage[t-1])/solution.resolution/solution.efficiency)
            flex = np.minimum(fill, solution.CPeak - flexible[t], solution.CPHP - solution.Charge[t] + solution.Discharge[t])

            fill -= flex
            flexible[t] += flex
            
    Resimulate(solution, flexible=flexible)
    # enforce resource limit
    if flexible.sum() - solution.Flex_res > 1: #allow tolerance
        flexible = np.maximum(flexible - solution.Spillage, 0) 
        flexible = flexible * min(1, solution.Flex_res / flexible.sum())
        Resimulate(solution, flexible=flexible)
        
    return flexible


def Analysis(solution):
    """Dispatch.Analysis(result.x)"""

    starttime = dt.datetime.now()
    print('Fill starts at', starttime)
    Flex = Fill(solution)
    endtime = dt.datetime.now()
    print('Fill took', endtime - starttime)

    np.savetxt(f'Results/Dispatch_Flexible{solution.scenario}.csv', Flex, fmt='%f', delimiter=',', newline='\n', header='Flexible energy resources')

    from Statistics import Information
    Information(solution.x)

    return True

if __name__ == '__main__':
    from Input import * 
    x = np.genfromtxt(f'Results/Optimisation_resultx{scenario}.csv', delimiter=',', dtype=float)
    
    Analysis(Solution(x))
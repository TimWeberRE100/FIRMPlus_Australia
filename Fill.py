import numpy as np 
from numba import njit
import datetime as dt

from Resimulation import Resimulate
from Interconnection import hvdc

perfect = np.array([0,1,3,6,10,15,21])


@njit 
def Fill(solution):
    flexible = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)
    Resimulate(solution, flexible=flexible)
    
    fill = np.zeros(solution.nodes, np.float64)
    for t in range(solution.intervals-1, -1, -1):
        if solution.MDeficit[t].sum() > 1e-6:
            flexible[t] = np.minimum(solution.MDeficit[t], solution.CPeak)
            if solution.MDeficit[t].sum() - flexible[t].sum() > 1e-6:
                _import, _export = solution.TImport[t].copy(), solution.TExport[t].copy()
                hvdc(solution, solution.MDeficit[t], solution.CPeak - flexible[t], 
                     solution.TImport[t], solution.TExport[t])
                flexible[t] += np.maximum(0, (_import + _export - solution.TImport[t] - solution.TExport[t]).sum(axis=0))
            fill += (solution.MDeficit[t]-flexible[t])/solution.efficiency
        if fill.sum() > 1e-6:
            # # simplified charging model
            fill = np.minimum(fill, (solution.CPHS - solution.MStorage[t-1])/solution.resolution/solution.efficiency)
            flex = np.minimum(fill, solution.CPeak - flexible[t], solution.CPHP - solution.MCharge[t] + solution.MDischarge[t])
            if fill.sum() - flex.sum() > 1e-6:
                _import, _export = solution.TImport[t].copy(), solution.TExport[t].copy()
                hvdc(solution, solution.MDeficit[t], solution.CPeak - flexible[t] - flex, 
                     solution.TImport[t], solution.TExport[t])
                flex += np.maximum(0, (_import + _export - solution.TImport[t] - solution.TExport[t]).sum(axis=0))
            fill -= flex
            flexible[t] += flex
    Resimulate(solution, flexible=flexible)

    # enforce resource limit
    if solution.MFlexible.sum() - solution.Flex_res > 1: #allow tolerance
        flexible = np.maximum(flexible - solution.MSpillage, 0) 
        flexible = flexible * min(1, solution.Flex_res / flexible.sum())
        Resimulate(solution, flexible=flexible)
        
    return solution.MFlexible


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
import numpy as np 
from numba import njit
import datetime as dt

from Simulation import Simulate, Resimulate
from Interconnection import hvdc


@njit 
def Fill(solution):
    solution.MFlexible = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)
    Resimulate(solution)
    
    fill = np.zeros(solution.nodes, np.float64)
    for t in range(solution.intervals-1, -1, -1):
        if solution.MDeficit[t].sum() > 1e-6:
            solution.MFlexible[t] = np.minimum(solution.MDeficit[t], solution.CPeak)
            if solution.MDeficit[t].sum() - solution.MFlexible[t].sum() > 1e-6:
                _import, _export = solution.TImport[t].copy(), solution.TExport[t].copy()
                hvdc(solution, solution.MDeficit[t], solution.CPeak - solution.MFlexible[t], 
                     solution.TImport[t], solution.TExport[t])
                solution.MFlexible[t] += np.maximum(0, (_import + _export - solution.TImport[t] - solution.TExport[t]).sum(axis=0))
            fill += (solution.MDeficit[t]-solution.MFlexible[t])/solution.efficiency
        if fill.sum() > 1e-6:
            # # simplified charging model
            fill = np.minimum(fill, (solution.CPHS - solution.MStorage[t-1])/solution.resolution/solution.efficiency)
            flex = np.minimum(np.minimum(fill, 
                              solution.CPeak - solution.MFlexible[t]),
                              solution.CPHP - solution.MCharge[t] + solution.MDischarge[t])
            if fill.sum() - flex.sum() > 1e-6:
                _import, _export = solution.TImport[t].copy(), solution.TExport[t].copy()
                hvdc(solution, solution.MDeficit[t], solution.CPeak - solution.MFlexible[t] - flex, 
                     solution.TImport[t], solution.TExport[t])
                flex += np.maximum(0, (_import + _export - solution.TImport[t] - solution.TExport[t]).sum(axis=0))
            fill -= flex
            solution.MFlexible[t] += flex

    Simulate(solution, False)

    # enforce resource limit
    if solution.MFlexible.sum() - solution.Flex_res > 1: #allow tolerance
        solution.MFlexible = np.maximum(solution.MFlexible - solution.MSpillage, 0) 
        solution.MFlexible = solution.MFlexible * min(1, solution.Flex_res / solution.MFlexible.sum())
        Simulate(solution, False)
        
    solution.TDC = (np.atleast_3d(solution.trans_tdc_mask).T*(solution.TImport + solution.TExport)).sum(axis=2)

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
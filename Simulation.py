# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:53:22 2024

@author: u6942852
"""

import numpy as np
from numba import njit


@njit
def Simulate(solution, flexible):
    """ 
    flexible = np.ones((intervals, nodes))*CPeak*1000; end=None; start=None 
    """
    
    solution.MNetload = solution.MLoad - solution.MPV - solution.MWind - solution.CBaseload - flexible

    solution.MDischarge = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)
    solution.MCharge = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)
    solution.MStorage = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)

    solution.MStorage[-1] = 0.5*solution.CPHS

    for t in range(solution.intervals):
        solution.MCharge[t] = np.minimum(
            np.minimum(
                -np.minimum(
                    0, 
                    solution.MNetload[t]
                    ), 
                solution.CPHP
                ), 
            (solution.CPHS - solution.MStorage[t-1]) / solution.efficiency / solution.resolution
            )
        solution.MDischarge[t] = np.minimum(
            np.minimum(
                np.maximum(
                    0, 
                    solution.MNetload[t]), 
                solution.CPHP
                ), 
            solution.MStorage[t-1] / solution.resolution
            )
        solution.MStorage[t] = solution.MStorage[t-1] +  solution.resolution * (
            solution.MCharge[t] * solution.efficiency - solution.MDischarge[t])

    solution.MFlexible = flexible
    solution.MSpillage = -np.minimum(0, solution.MNetload + solution.MCharge)
    solution.MDeficit = np.maximum(0, solution.MNetload - solution.MDischarge)
    
    return solution.MDeficit

        
    
    
    
    
    
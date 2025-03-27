# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:02:45 2025

@author: u6942852
"""


import numpy as np
from numba import njit

from Interconnection import hvdc

from Simulation import Simulate


@njit
def Resimulate(solution, flexible):
    """ 
    flexible = np.ones((intervals, nodes))*CPeak*1000
    flexible = np.zeros((intervals, nodes))
    """
    # These hold a matrix of import and export by node and line for each time
    solution.TImport = np.zeros((solution.intervals, solution.nhvdc, solution.nodes), dtype = np.float64)
    solution.TExport = np.zeros((solution.intervals, solution.nhvdc, solution.nodes), dtype = np.float64)
    
    # Get first approximation without transmission
    Simulate(solution, flexible)
    
    fill = np.zeros(solution.nodes, np.float64) # Deficit accumulator
    storage_adjuster = np.zeros(solution.nodes, np.float64)
    
    for t in range(solution.intervals-1, -1, -1):
        # If there is a deficit try to fill with transmission
        if solution.MDeficit[t].sum() > 1e-6:
            # Surplus is spillage + charging - dischargeable storage - discharging 
            Surplus = np.maximum(0, 
                solution.MSpillage[t] + solution.MCharge[t] + 
                np.minimum(solution.CPHP, solution.MStorage[t-1] / solution.resolution - storage_adjuster) - solution.MDischarge[t]
                )
            
            # If no surplus, no need to waste time here
            if Surplus.sum() > 1e-6: 
                # Calculate transmission flows (in-place)
                hvdc(solution, solution.MDeficit[t], Surplus, solution.TImport[t], solution.TExport[t])
                
                # Recalculate charging behaviour as normal but adjusting netload for import/export
                solution.MCharge[t] = np.minimum(
                    np.minimum(
                        - np.minimum(0, solution.MNetload[t] - (solution.TImport[t] + solution.TExport[t]).sum(axis=0)), 
                        solution.CPHP), 
                    (solution.CPHS - solution.MStorage[t-1]) / solution.efficiency / solution.resolution
                    )
                solution.MDischarge[t] = np.minimum(
                    np.minimum(
                        np.maximum(0,solution.MNetload[t] - (solution.TImport[t] + solution.TExport[t]).sum(axis=0)), 
                        solution.CPHP), 
                    solution.MStorage[t-1] / solution.resolution
                    )
                # Recalculate spillage 
                solution.MSpillage[t] = -np.minimum(0, solution.MNetload[t] - (solution.TImport[t] + solution.TExport[t]).sum(axis=0) + solution.MCharge[t])            
                # Deficit does not need to be recalculated as hvdc() is in-place
                
            # accumulate deficit + charging inefficiency
            if solution.MDeficit[t].sum() > 1e-6:
                fill += solution.MDeficit[t] / solution.efficiency 
                
        # Fill accumulated deficits
        if fill.sum() > 1e-6:
            # Cap accumulator if storage full 
            #TODO: improve charging model to cap a node importing when storage is full as in Fill
            fill = np.minimum(fill, (solution.CPHS - solution.MStorage[t-1])/solution.resolution/solution.efficiency)
            
            # Surplus is spillage + charging - dischargeable storage - discharging 
            Surplus = np.maximum(0, 
                solution.MSpillage[t] + solution.MCharge[t] + 
                np.minimum(solution.CPHP, solution.MStorage[t-1] / solution.resolution - storage_adjuster) - solution.MDischarge[t]
                
                )
            
            # If no surplus, no need to waste time here
            if Surplus.sum() > 1e-6: 
                # Calculate transmission flows (in-place)
                hvdc(solution, fill, Surplus, solution.TImport[t], solution.TExport[t])
                
                # Recalculate charging behaviour as normal but adjusting netload for import/export
                solution.MCharge[t] = np.minimum(
                    np.minimum(
                        - np.minimum(0, solution.MNetload[t] - (solution.TImport[t] + solution.TExport[t]).sum(axis=0)), 
                        solution.CPHP), 
                    (solution.CPHS - solution.MStorage[t-1]) / solution.efficiency / solution.resolution
                    )
                solution.MDischarge[t] = np.minimum(
                    np.minimum(
                        np.maximum(0,solution.MNetload[t] - (solution.TImport[t] + solution.TExport[t]).sum(axis=0)), 
                        solution.CPHP), 
                    solution.MStorage[t-1] / solution.resolution
                    )
                # Recalculate spillage 
                solution.MSpillage[t] = -np.minimum(0, solution.MNetload[t] - (solution.TImport[t] + solution.TExport[t]).sum(axis=0) + solution.MCharge[t])
                # fill does not need to be updated as hvdc() is in-place
            
        # Update SOC 
        MStorage = solution.MStorage[t-1] + solution.resolution * (solution.MCharge[t] * solution.efficiency - solution.MDischarge[t])
        storage_adjuster += MStorage - solution.MStorage[t] 
        solution.MStorage[t-1] += storage_adjuster 

        
    Simulate(solution, flexible)
    
    solution.TDC = (np.atleast_3d(solution.trans_tdc_mask).T*(solution.TImport + solution.TExport)).sum(axis=2)

    return solution.MDeficit

            
                
    

            
    
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:53:22 2024

@author: u6942852
"""

import numpy as np
from numba import njit

from Interconnection import Interconnection

@njit
def Simulate(solution):
    TransmissionSimulate(solution)
    if solution.MDeficit.sum() <= 1e-6:
        return 
    
    fill = np.zeros(solution.nodes, dtype=np.float64)
    #timestep backwards
    for t in range(solution.intervals-1, -1, -1):
        if solution.MDeficit[t].sum() > 1e-6:
            # accumulate deficit + storage efficiency 
            fill += solution.MDeficit[t]/solution.efficiency
        if fill.sum() > 1e-6 and solution.MSpillage[t].sum() > 1e-6:
            # cap fill by storage capacity
            fill = np.minimum(fill, (solution.CPHS - solution.MStorage[t-1])/solution.resolution/solution.efficiency)
            # meet fill with neighbours' spillage - don't draw down power as this affects future SOC
            Interconnection(solution, fill, solution.MSpillage[t], solution.TImport[t], solution.TExport[t])
            # fill adjusted in-place
    # fix storage traces
    BasicSimulate(solution)
    if solution.MDeficit.sum() <= 1e-6:
        return 

    fill = np.zeros(solution.nodes, np.float64)
    for t in range(solution.intervals-1, -1, -1):
        # timestep backwards
        if solution.MDeficit[t].sum() > 1e-6:
            # meet deficits just-in-time with flex
            solution.MFlexible[t] = np.minimum(solution.MDeficit[t], solution.CPeak)
            # if remaining deficits:
            if solution.MDeficit[t].sum() - solution.MFlexible[t].sum() > 1e-6:
                # original import/export
                _import, _export = solution.TImport[t].copy(), solution.TExport[t].copy()
                # meet deficits just-in-time by importiing flex from neighbours
                Interconnection(solution, solution.MDeficit[t], solution.CPeak - solution.MFlexible[t], 
                     solution.TImport[t], solution.TExport[t])
                # flexible += iexports from neighbours
                solution.MFlexible[t] += np.maximum(0, (_import + _export - solution.TImport[t] - solution.TExport[t]).sum(axis=0))
            # accumulate remaing deficits
            fill += (solution.MDeficit[t]-solution.MFlexible[t])/solution.efficiency
        if fill.sum() > 1e-6:
            # # simplified charging model
            fill = np.minimum(fill, (solution.CPHS - solution.MStorage[t-1])/solution.resolution/solution.efficiency)
            flex = np.minimum(np.minimum(fill, 
                              solution.CPeak - solution.MFlexible[t]),
                              solution.CPHP - solution.MCharge[t] + solution.MDischarge[t])
            if fill.sum() - flex.sum() > 1e-6:
                _import, _export = solution.TImport[t].copy(), solution.TExport[t].copy()
                Interconnection(solution, fill-flex, solution.CPeak - solution.MFlexible[t] - flex, 
                     solution.TImport[t], solution.TExport[t])
                flex += np.maximum(0, (_import + _export - solution.TImport[t] - solution.TExport[t]).sum(axis=0))
            fill -= flex
            solution.MFlexible[t] += flex

    BasicSimulate(solution)
    
    solution.TDC = (np.atleast_3d(solution.trans_mask).T*(solution.TImport + solution.TExport)).sum(axis=2)
    
@njit
def TransmissionSimulate(solution):
    for t in range(solution.intervals):
        # storage operation 
        StorageBehaviourt(solution, t)
        Imbalancet(solution, t)
        
        # fill deficits from spilled power
        if solution.MDeficit[t].sum() > 1e-6:
            if solution.MSpillage[t].sum() > 1e-6: 
                Interconnection(solution, solution.MDeficit[t], solution.MSpillage[t], solution.TImport[t], solution.TExport[t])
                # update storage behaviour  
                StorageBehaviourt(solution, t)
                # Imbalancet(solution, t) # updated inplace by Interconnection
                
        # fill deficits by drawing down neighbours' storage reserves
        if solution.MDeficit[t].sum() > 1e-6:
            break
            Surplus = np.maximum(0, solution.MSpillage[t] + solution.MCharge[t] + 
                np.minimum(solution.CPHP, solution.MStorage[t-1] / solution.resolution) - solution.MDischarge[t])
            if Surplus.sum() > 1e-6: 
                Interconnection(solution, solution.MDeficit[t], Surplus, solution.TImport[t], solution.TExport[t])
                # update storage behaviour  
                StorageBehaviourt(solution, t)
                # update deficit/spillage
                Imbalancet(solution, t) 
        
        # export as much spillage as possible
        if solution.MSpillage[t].sum() > 1e-6:
            # This is surplus charging capacity, not export capacity
            Surplus = np.minimum(solution.CPHP - solution.MCharge[t] + solution.MDischarge[t], # charge capacity
                       (solution.CPHS-solution.MStorage[t-1])/solution.resolution/solution.efficiency) # energy constraint 
            Interconnection(solution, Surplus, solution.MSpillage[t], solution.TImport[t], solution.TExport[t])
            # update storage behaviour  
            StorageBehaviourt(solution, t)
            # update deficit/spillage
            Imbalancet(solution, t)
        
        UpdateSOCt(solution, t)
    solution.TDC = (np.atleast_3d(solution.trans_mask).T*(solution.TImport + solution.TExport)).sum(axis=2)
    
@njit
def BasicSimulate(solution):
    for t in range(solution.intervals):
        # storage operation
        StorageBehaviourt(solution, t) 
        UpdateSOCt(solution, t)
    # Deficit and Spillage
    Imbalance(solution)


@njit 
def UpdateSOCt(solution, t):
    # Previous energy + charge - discharge
    solution.MStorage[t] = solution.MStorage[t-1] +  solution.resolution * (
        solution.MCharge[t] * solution.efficiency - solution.MDischarge[t])

@njit
def StorageBehaviourt(solution, t):
    solution.MCharge[t] = np.minimum(np.minimum(
             #available/required power = demand - generation - imports 
            -np.minimum(0, solution.MNetload[t] - solution.MFlexible[t] - (solution.TImport[t] + solution.TExport[t]).sum(axis=0)), 
            solution.CPHP), # storage power constraint
            (solution.CPHS - solution.MStorage[t-1]) / solution.efficiency / solution.resolution) #storage energy constraint
    solution.MDischarge[t] = np.minimum(np.minimum(
             #available/required power = demand - generation - imports 
            np.maximum(0, solution.MNetload[t] - solution.MFlexible[t] - (solution.TImport[t] + solution.TExport[t]).sum(axis=0)), 
            solution.CPHP), # storage power constraint
        solution.MStorage[t-1] / solution.resolution) # storage energy constraint
  
@njit
def Imbalancet(solution, t):
    solution.MDeficit[t] = np.maximum(0, 
        solution.MNetload[t] - solution.MFlexible[t] - (solution.TImport[t] + solution.TExport[t]).sum(axis=0) + solution.MCharge[t] - solution.MDischarge[t])
    solution.MSpillage[t] = -np.minimum(0, 
        solution.MNetload[t] - solution.MFlexible[t] - (solution.TImport[t] + solution.TExport[t]).sum(axis=0) + solution.MCharge[t] - solution.MDischarge[t])

@njit
def Imbalance(solution):
    # deficit is a positive netload not met by storage or imports
    solution.MDeficit = np.maximum(0, 
        solution.MNetload - solution.MFlexible - (solution.TImport + solution.TExport).sum(axis=1) + solution.MCharge - solution.MDischarge)
    # spillage is a negative netload not absorbed by storage or exports  
    solution.MSpillage = -np.minimum(0, 
        solution.MNetload - solution.MFlexible - (solution.TImport + solution.TExport).sum(axis=1) + solution.MCharge - solution.MDischarge)


    
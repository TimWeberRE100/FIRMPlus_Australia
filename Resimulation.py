# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:02:45 2025

@author: u6942852
"""


import numpy as np
from numba import njit

from Interconnection import hvdc

perfect = np.array([0,1,3,6,10,15,21])

from Simulation import Simulate


@njit
def Resimulate(solution, flexible):
    """ 
    flexible = np.ones((intervals, nodes))*CPeak*1000; end=None; start=None 
    """
    network = solution.network
    trans_tdc_mask = solution.trans_tdc_mask
    networksteps = np.where(perfect == network.shape[2])[0][0]
    
    Simulate(solution, flexible, False)
    
    Transmission = np.zeros((solution.intervals, solution.nhvdc, solution.nodes), dtype = np.float64)
    
    fill = np.zeros(solution.nodes, np.float64)
    for t in range(solution.intervals-1, -1, -1):
        if solution.Deficit[t].sum() > 1e-6:
            Surplus = np.maximum(0, 
                solution.Spillage[t] + solution.Charge[t] + 
                np.minimum(solution.CPHP, solution.Storage[t-1] / solution.resolution) - solution.Discharge[t]
                )
            
            if Surplus.sum() > 1e-6: 
                Transmission[t] = hvdc(solution.Deficit[t], Surplus, solution.CHVDC, network, networksteps, 
                                     np.maximum(0, Transmission[t]), np.minimum(0, Transmission[t]))
                
                solution.Charge[t] = np.minimum(
                    np.minimum(
                        - np.minimum(0, solution.Netload[t] - Transmission[t].sum(axis=0)), 
                        solution.CPHP), 
                    (solution.CPHS - solution.Storage[t-1]) / solution.efficiency / solution.resolution
                    )
                solution.Discharge[t] = np.minimum(
                    np.minimum(
                        np.maximum(0,solution.Netload[t] - Transmission[t].sum(axis=0)), 
                        solution.CPHP), 
                    solution.Storage[t-1] / solution.resolution
                    )
                solution.Spillage[t] = -np.minimum(0, solution.Netload[t] - Transmission[t].sum(axis=0) + solution.Charge[t])            
            
            if solution.Deficit[t].sum() > 1e-6:
                fill += solution.Deficit[t] / solution.efficiency 
                
        if fill.sum() > 1e-6:
            #TODO: add simplified charging model to cap a node importing when storage is full as in Fill
            fill = np.minimum(fill, (solution.CPHS - solution.Storage[t-1])/solution.resolution/solution.efficiency)

            Surplus = np.maximum(0, 
                solution.Spillage[t] + solution.Charge[t] + 
                np.minimum(solution.CPHP, solution.Storage[t-1] / solution.resolution) - solution.Discharge[t]
                )
            
            if Surplus.sum() > 1e-6: 
                Transmission[t] = hvdc(fill, Surplus, solution.CHVDC, network, networksteps, 
                                     np.maximum(0, Transmission[t]), np.minimum(0, Transmission[t]))
                
                solution.Charge[t] = np.minimum(
                    np.minimum(
                        - np.minimum(0, solution.Netload[t] - Transmission[t].sum(axis=0)), 
                        solution.CPHP), 
                    (solution.CPHS - solution.Storage[t-1]) / solution.efficiency / solution.resolution
                    )
                solution.Discharge[t] = np.minimum(
                    np.minimum(
                        np.maximum(0,solution.Netload[t] - Transmission[t].sum(axis=0)), 
                        solution.CPHP), 
                    solution.Storage[t-1] / solution.resolution
                    )
                solution.Spillage[t] = -np.minimum(0, solution.Netload[t] - Transmission[t].sum(axis=0) + solution.Charge[t])
        
        solution.Storage[t] = solution.Storage[t-1] - solution.resolution * (solution.Discharge[t]
                               + solution.Charge[t] * solution.efficiency)

    ImpExp = Transmission.sum(axis=1)
    
    # Already calcualted in for loop above
    # solution.Deficit = np.maximum(0, solution.Netload - ImpExp - solution.Discharge)
    # solution.Spillage = -1 * np.minimum(0, solution.Netload - ImpExp + solution.Charge)
    
    solution.Import = np.maximum(0, ImpExp)
    solution.Export = -1 * np.minimum(0, ImpExp)
    solution.TDC = (np.atleast_3d(trans_tdc_mask).T*Transmission).sum(axis=2)

    return solution.Deficit

            
                
    

            
    
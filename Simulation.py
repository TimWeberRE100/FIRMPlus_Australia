# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:53:22 2024

@author: u6942852
"""

import numpy as np
from numba import njit

from Interconnection import hvdc

perfect = np.array([0,1,3,6,10,15,21])


@njit
def Simulate(solution, flexible, transmission=True):
    """ 
    flexible = np.ones((intervals, nodes))*CPeak*1000; end=None; start=None 
    """
    network = solution.network
    trans_tdc_mask = solution.trans_tdc_mask
    networksteps = np.where(perfect == network.shape[2])[0][0]
    
    Netload = (solution.MLoad - solution.GPV - solution.GWind - solution.GBaseload)
    Netload -= flexible
    

    Discharge = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)
    Charge = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)
    Storage = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)
    Deficit = np.zeros((solution.intervals, solution.nodes), dtype=np.float64)
    Transmission = np.zeros((solution.intervals, solution.nhvdc, solution.nodes), dtype = np.float64)

    Storage[-1] = 0.5*solution.CPHS

    for t in range(solution.intervals):
        Netloadt = Netload[t]

        Charge[t] = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), solution.CPHP), (solution.CPHS - Storage[t-1]) / solution.efficiency / solution.resolution)
        Discharge[t] = np.minimum(np.minimum(np.maximum(0, Netloadt), solution.CPHP), Storage[t-1] / solution.resolution)
        Deficitt = np.maximum(Netloadt - Discharge[t] ,0)

        if Deficitt.sum() > 1e-6 and transmission is True:
            # raise KeyboardInterrupt
            # Fill deficits with transmission allowing drawing down from neighbours battery reserves
            Fillt = np.maximum(Netloadt - Discharge[t], 0)
            Surplust = -1 * np.minimum(0, Netloadt + Charge[t]) + np.minimum(solution.CPHP, Storage[t-1] / solution.resolution)

            Transmission[t] = hvdc(Fillt, Surplust, solution.CHVDC, network, networksteps, 
                                 np.maximum(0, Transmission[t]), np.minimum(0, Transmission[t]))
            
            Netloadt = Netload[t] - Transmission[t].sum(axis=0)
            Charge[t] = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), solution.CPHP), (solution.CPHS - Storage[t-1]) / solution.efficiency / solution.resolution)
            Discharge[t] = np.minimum(np.minimum(np.maximum(0, Netloadt), solution.CPHP), Storage[t-1] / solution.resolution)

        # =============================================================================
        # TODO: If deficit Go back in time and discharge batteries 
        # This will be extemely computationally intensive
        # =============================================================================
        
        Surplust = -1 * np.minimum(0, Netloadt + Charge[t]) 
        if Surplust.sum() > 1e-6 and transmission is True:
            # raise KeyboardInterrupt
            # Distribute surplus energy with transmission to areas with spare charging capacity
            Fillt = (np.maximum(0, Netloadt) #load
                        + np.minimum(solution.CPHP, (solution.CPHS - Storage[t-1]) / solution.efficiency / solution.resolution) #full charging capacity
                        - Charge[t]) #charge capacity already in use

            Transmission[t] = hvdc(Fillt, Surplust, solution.CHVDC, network, networksteps,
                                 np.maximum(0, Transmission[t]), np.minimum(0, Transmission[t]))
            
            Netloadt = Netload[t] - Transmission[t].sum(axis=0)
            Charge[t] = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), solution.CPHP), (solution.CPHS - Storage[t-1]) / solution.efficiency / solution.resolution)
            Discharge[t] = np.minimum(np.minimum(np.maximum(0, Netloadt), solution.CPHP), Storage[t-1] / solution.resolution)

        Storage[t] = Storage[t-1] - Discharge[t] * solution.resolution + Charge[t] * solution.resolution * solution.efficiency
        
        
    ImpExp = Transmission.sum(axis=1)
    
    Deficit = np.maximum(0, Netload - ImpExp - Discharge)
    Spillage = -1 * np.minimum(0, Netload - ImpExp + Charge)

    solution.flexible = flexible
    solution.Spillage = Spillage
    solution.Charge = Charge
    solution.Discharge = Discharge
    solution.Storage = Storage
    solution.Deficit = Deficit
    solution.Import = np.maximum(0, ImpExp)
    solution.Export = -1 * np.minimum(0, ImpExp)
    solution.TDC = (np.atleast_3d(trans_tdc_mask).T*Transmission).sum(axis=2)
    solution.Netload = Netload
    
    return Deficit

        
    
    
    
    
    
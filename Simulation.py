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
def Reliability(solution, flexible, start=None, end=None):
    """ 
    flexible = np.ones((intervals, nodes))*CPeak*1000; end=None; start=None 
    """
    network = solution.network
    trans_tdc_mask = solution.trans_tdc_mask
    networksteps = np.where(perfect == network.shape[2])[0][0]
    
    Netload = (solution.MLoad - solution.GPV - solution.GWind - solution.GBaseload)[start:end]
    Netload -= flexible
    
    shape2d = intervals, nodes = len(Netload), solution.nodes

    Pcapacity = solution.CPHP * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # S-CPHS(j), GWh to MWh
    Hcapacity = solution.CHVDC * 1000 # GW to MW
    nhvdc = len(solution.CHVDC)
    efficiency, resolution = solution.efficiency, solution.resolution 

    Discharge = np.zeros(shape2d, dtype=np.float64)
    Charge = np.zeros(shape2d, dtype=np.float64)
    Storage = np.zeros(shape2d, dtype=np.float64)
    Deficit = np.zeros(shape2d, dtype=np.float64)
    Transmission = np.zeros((intervals, nhvdc, nodes), dtype = np.float64)

    Storaget_1 = 0.5*Scapacity

    for t in range(intervals):
        Netloadt = Netload[t]

        Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
        Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
        Deficitt = np.maximum(Netloadt - Discharget ,0)

        Transmissiont=np.zeros((nhvdc, nodes), dtype=np.float64)
        
        # # The following provides a relatively minor benefit to cost for increased computation time
        # if Deficitt.sum() > 1e-6:
        #     # Fill deficits with transmission without drawing down from battery reserves
        #     Fillt = np.maximum(Netloadt - Discharget, 0)
        #     Surplust = -1 * np.minimum(0, Netloadt + Charget) 
            
        #     Transmissiont = hvdc(Fillt, Surplust, Transmissiont, Hcapacity, network, networksteps)
            
        #     Netloadt = Netload[t] - Transmissiont.sum(axis=0)
        #     Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
        #     Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
        #     Deficitt = np.maximum(Netloadt - Discharget, 0)
    
        if Deficitt.sum() > 1e-6:
            # raise KeyboardInterrupt
            # Fill deficits with transmission allowing drawing down from neighbours battery reserves
            Fillt = np.maximum(Netloadt - Discharget, 0)
            Surplust = -1 * np.minimum(0, Netloadt + Charget) + np.minimum(Pcapacity, Storaget_1 / resolution)

            Transmissiont = hvdc(Fillt, Surplust, Hcapacity, network, networksteps, 
                                 np.maximum(0, Transmissiont), np.minimum(0, Transmissiont))
            
            Netloadt = Netload[t] - Transmissiont.sum(axis=0)
            Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
            Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)

        # =============================================================================
        # TODO: If deficit Go back in time and discharge batteries 
        # This will be extemely computationally intensive
        # =============================================================================
        
        Surplust = -1 * np.minimum(0, Netloadt + Charget) 
        if Surplust.sum() > 1e-6:
            # raise KeyboardInterrupt
            # Distribute surplus energy with transmission to areas with spare charging capacity
            Fillt = (np.maximum(0, Netloadt) #load
                        + np.minimum(Pcapacity, (Scapacity - Storaget_1) / efficiency / resolution) #full charging capacity
                        - Charget) #charge capacity already in use

            Transmissiont = hvdc(Fillt, Surplust, Hcapacity, network, networksteps,
                                 np.maximum(0, Transmissiont), np.minimum(0, Transmissiont))
            
            Netloadt = Netload[t] - Transmissiont.sum(axis=0)
            Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
            Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)

        Storaget = Storaget_1 - Discharget * resolution + Charget * resolution * efficiency
        Storaget_1 = Storaget.copy()
        
        Discharge[t] = Discharget
        Charge[t] = Charget
        Storage[t] = Storaget
        Transmission[t] = Transmissiont
        
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
    
    return Deficit


        
    
    
    
    
    
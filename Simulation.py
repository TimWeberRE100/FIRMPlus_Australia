# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:53:22 2024

@author: u6942852
"""

import numpy as np
from numba import njit

perfect = np.array([0,1,3,6,10,15,21])


@njit
def Reliability(solution, flexible, start=None, end=None):
    """ flexible = np.ones((intervals, nodes))*CPeak*1000 """

    PVl_int, Windl_int = solution.PVl_int, solution.Windl_int
    network = solution.network
    networksteps = np.where(perfect == network.shape[1])[0][0]
    
    if start is None and end is None:
        shape2d = intervals, nodes = solution.intervals, solution.nodes
        Netload = np.empty(shape2d, dtype=np.float64)
        for i, j in enumerate(solution.Nodel_int):
            Netload[:, i] = (solution.MLoad[:, i] 
                             - solution.GPV[:, PVl_int==j].sum(axis=1)
                             - solution.GWind[:, Windl_int==j].sum(axis=1)
                             - solution.GBaseload[:, i]
                             )
    else: 
        shape2d = intervals, nodes = end-start, solution.nodes
        Netload = np.empty(shape2d, dtype=np.float64)
        for i, j in enumerate(solution.Nodel_int):
            Netload[:, i] = (solution.MLoad[:, i] 
                             - solution.GPV[:, PVl_int==j].sum(axis=1)
                             - solution.GWind[:, Windl_int==j].sum(axis=1)
                             - solution.GBaseload[:, i]
                             )[start:end]
    Netload -= flexible

    Pcapacity = solution.CPHP * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # S-CPHS(j), GWh to MWh
    Hcapacity = solution.CHVDC * 1000
    nhvdc = len(solution.CHVDC)
    efficiency, resolution = solution.efficiency, solution.resolution 

    Discharge = np.zeros(shape2d, dtype=np.float64)
    Charge = np.zeros(shape2d, dtype=np.float64)
    Storage = np.zeros(shape2d, dtype=np.float64)
    Deficit = np.zeros(shape2d, dtype=np.float64)
    Transmission = np.zeros(shape2d, dtype=np.float64)
    TDC = np.zeros((intervals, nhvdc))

    for t in range(intervals):
        Netloadt = Netload[t]
        Storaget_1 = Storage[t-1,:] if t>0 else 0.5*Scapacity

        Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
        Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
        Deficitt = np.maximum(Netloadt - Discharget ,0)

        Transmissiont = np.zeros((nhvdc, nodes), dtype=np.float64)
        
        # if Deficitt.sum() > 1e-6:
        #     # Fill deficits with transmission without drawing down from battery reserves
        #     fill_req = np.maximum(Netloadt - Discharget, 0)
        #     Surplust = -1 * np.minimum(0, Netloadt + Charget) 
            
        #     Transmissiont = hvdc(fill_req, Surplust, Transmissiont, Hcapacity, network, networksteps)
            
        #     Netloadt = Netload[t] - Transmissiont.sum(axis=0)
        #     Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
        #     Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
        #     Deficitt = np.maximum(Netloadt - Discharget, 0)
    
        if Deficitt.sum() > 1e-6: 
            # Fill deficits with transmission by drawing down from battery reserves
            fill_req = np.maximum(Netloadt - Discharget, 0)
            Surplust = np.minimum(Pcapacity, Storaget_1 / resolution)
            
            Transmissiont = hvdc(fill_req, Surplust, Transmissiont, Hcapacity, network, networksteps)
            
            Netloadt = Netload[t] - Transmissiont.sum(axis=0)
            Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
            Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)

        # =============================================================================
        # TODO: If deficit Go back in time and discharge batteries 
        # This may make the time a fair bit longer
        # =============================================================================
        
        Surplust = -1 * np.minimum(0, Netloadt + Charget) 
        if Surplust.sum() > 1e-6:
            # Distribute surplus energy with transmission to areas with spare charging capacity
            fill_req = (np.maximum(0, Netloadt) #load
                        + np.minimum(Pcapacity, (Scapacity - Storaget_1) / efficiency / resolution) #full charging capacity
                        - Charget) #charge capacity already in use

            Transmissiont = hvdc(
                fill_req, Surplust, Transmissiont, Hcapacity, network, networksteps)
            
            Netloadt = Netload[t] - Transmissiont.sum(axis=0)
            Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
            Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)

        Storaget = Storaget_1 - Discharget * resolution + Charget * resolution * efficiency
        
        Discharge[t] = Discharget
        Charge[t] = Charget
        Storage[t] = Storaget
        Transmission[t] = Transmissiont.sum(axis=0)
        TDC[t] = np.maximum(0, Transmissiont).sum(axis=1)
        
    Deficit = np.maximum(Netload - Transmission - Discharge, np.zeros_like(Netload))
    Spillage = -1 * np.minimum(Netload + Charge, np.zeros_like(Netload))

    solution.flexible = flexible
    solution.Spillage = Spillage
    solution.Charge = Charge
    solution.Discharge = Discharge
    solution.Storage = Storage
    solution.Deficit = Deficit
    solution.Transmission = Transmission
    solution.TDC = TDC
    
    return Deficit

@njit
def hvdc(fill_req, Surplust, Transmissiont, Hcapacity, network, networksteps):
    for n, net in enumerate(network):
        if fill_req[n] == 0:
            continue
        
        for leg in range(networksteps):
            donors = net[perfect[leg]:perfect[leg+1],:, :]
            donors, donor_lines = donors[:,:,0], donors[:,:,1]

            valid_mask = donors[-1] != -1
            if np.prod(~valid_mask):
                break
            donor_lines = donor_lines[:, valid_mask]
            donors = donors[:, valid_mask]

            ndonors = valid_mask.sum()
            donors = np.concatenate((n*np.ones((1, ndonors), np.int64), donors))
            d=0
            while fill_req[n] > 0 and d < ndonors:
                #TODO at the moment surplus is taken from first zone, and second isn't 
                #     touched unless necessary. Would be nice to take from all evenly.
                donor_line_cap = np.inf 
                for l in donor_lines[:,d]:
                    # transmission cap is minimum of capacities of lines involved
                    donor_line_cap = min(
                        donor_line_cap, 
                        Hcapacity[l] - np.maximum(0, Transmissiont[l, :]).sum()
                        )
                                           
                _transmission = max(0, min(donor_line_cap, Surplust[donors[-1, d]], fill_req[n]))
                
                for step in range(leg+1): 
                    Transmissiont[donor_lines[step, d], donors[step, d]] += _transmission
                    Transmissiont[donor_lines[step, d], donors[step+1, d]] -= _transmission

                fill_req[n] -= _transmission
                Surplust[donors[-1, d]] -= _transmission
                d+=1
                
            if fill_req[n] == 0:
                break
                
    return Transmissiont


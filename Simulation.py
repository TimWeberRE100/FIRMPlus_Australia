# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:53:22 2024

@author: u6942852
"""

import numpy as np
from numba import njit, int64

@njit
def Reliability(solution, flexible, start=None, end=None):
    PVl_int, Windl_int = solution.PVl_int, solution.Windl_int
    network = solution.network
    perfect = {0:0, 1:1, 2:3, 3:6, 4:10, 5:15, 6:21} #that's more than enough for now
    unperfect = {v:k for k,v in perfect.items()}
    networksteps = unperfect[network.shape[1]]
    
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
    
    """ flexible = np.ones((intervals, nodes))*CPeak*1000 """

    Pcapacity = solution.CPHP * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # S-CPHS(j), GWh to MWh
    Hcapacity = solution.CHVDC * 1000
    nhvdc = len(solution.CHVDC)
    efficiency, resolution = solution.efficiency, solution.resolution 

    Discharge = np.zeros(shape2d, dtype=np.float64)
    Charge = np.zeros(shape2d, dtype=np.float64)
    Storage = np.zeros(shape2d, dtype=np.float64)
    Deficit = np.zeros(shape2d, dtype=np.float64)
    Import = np.zeros(shape2d, dtype=np.float64)
    Export = np.zeros(shape2d, dtype=np.float64)
    TDC = np.zeros((intervals, nhvdc))

    for t in range(intervals):
        Netloadt = Netload[t]
        Storaget_1 = Storage[t-1,:] if t>0 else 0.5*Scapacity

        Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
        Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
        Deficitt = np.maximum(Netloadt - Discharget ,0)

        Surplust = -1 * np.minimum(0, Netloadt + Charget) 
       
        Importt = np.zeros((nhvdc, nodes), dtype=np.float64)
        Exportt = np.zeros((nhvdc, nodes), dtype=np.float64) 
        
        if Deficitt.sum() > 1e-6:
            # Fill deficits with transmission without drawing down from battery reserves
            fill_req = np.maximum(Netloadt - Discharget, 0)
            
            fill_req, Importt, Exportt = hvdc_control_new(
                fill_req, Surplust, Importt, Exportt, Hcapacity, network, networksteps, perfect)
            
            Netloadt = Netload[t] - Importt.sum(axis=0) + Exportt.sum(axis=0)
            Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
            Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
            Deficitt = np.maximum(Netloadt - Discharget ,0)
    
        if Deficitt.sum() > 1e-6: 
            # Fill deficits with transmission by drawing down from battery reserves
            fill_req = np.maximum(Netloadt - Discharget, 0)
            Surplust = np.minimum(Pcapacity, Storaget_1 / resolution)
            
            fill_req, Importt, Exportt = hvdc_control_new(
                fill_req, Surplust, Importt, Exportt, Hcapacity, network, networksteps, perfect)
            
            Netloadt = Netload[t] - Importt.sum(axis=0) + Exportt.sum(axis=0)
            Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
            Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)

        # =============================================================================
        # To Do: If deficit Go back in time and discharge batteries 
        # This may make the time a fair bit longer
        # =============================================================================
        
        Surplust = -1 * np.minimum(0, Netloadt + Charget) 
        if Surplust.sum() > 1e-6:
            # Distribute surplus energy with transmission to areas with spare charging capacity
            fill_req = (np.maximum(0, Netloadt) #load
                        + np.minimum(Pcapacity, (Scapacity - Storaget_1) / efficiency / resolution) #real charging capacity
                        - Charget) #charge capacity in use

            fill_req, Importt, Exportt = hvdc_control_new(
                fill_req, Surplust, Importt, Exportt, Hcapacity, network, networksteps, perfect)
            
            Netloadt = Netload[t] - Importt.sum(axis=0) + Exportt.sum(axis=0)
            Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
            Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)

        Storaget = Storaget_1 - Discharget * resolution + Charget * resolution * efficiency
        
        Discharge[t] = Discharget
        Charge[t] = Charget
        Storage[t] = Storaget
        Import[t] = Importt.sum(axis=0)
        Export[t] = Exportt.sum(axis=0)
        # assert (Exportt.sum(axis=1) - Importt.sum(axis=1)).sum() < 0.1
        TDC[t] = Exportt.sum(axis=1)
        
    Deficit = np.maximum(Netload - Import + Export - Discharge, np.zeros_like(Netload))
    Spillage = -1 * np.minimum(Netload + Charge, np.zeros_like(Netload))

    solution.flexible = flexible
    solution.Spillage = Spillage
    solution.Charge = Charge
    solution.Discharge = Discharge
    solution.Storage = Storage
    solution.Deficit = Deficit
    solution.Import = Import
    solution.Export = Export
    solution.TDC = TDC
    
    return Deficit

@njit
def hvdc_control_new(fill_req, Surplust, Importt, Exportt, Hcapacity, network, networksteps, perfect):
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
            fullpaths = np.concatenate((n*np.ones((1, ndonors), np.int64), donors))
            _rec=0
            d=0
            while fill_req[n] > 0 and _rec < (ndonors**2):
                
                donor_line_cap = np.inf 
                for l in donor_lines[:,d]:
                    # transmission cap is minimum of capacities of lines involved
                    donor_line_cap = np.minimum(
                        donor_line_cap, 
                        Hcapacity[l] - Exportt[l, :].sum()
                        )
                    
                _available = np.maximum(
                    0, 
                    np.minimum(
                        donor_line_cap,
                        np.minimum(
                            Surplust[donors[-1, d]], #energy availability
                            fill_req[n] # energy need
                            )
                        )
                    )
                
                if _available == 0:
                    _rec += 1 
                    continue
                
                for step in range(leg+1): 
                    Importt[donor_lines[step, d], fullpaths[step, d]] += _available
                    Exportt[donor_lines[step, d], fullpaths[step+1, d]] += _available

                _rec += 1
                d = (d+1)%ndonors
                fill_req[n] -= _available
                Surplust[donors[-1, d]] -= _available
                
            if fill_req[n] == 0:
                break
                
    return fill_req, Importt, Exportt 
    
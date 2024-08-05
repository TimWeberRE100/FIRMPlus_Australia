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
    """ flexible = np.ones((intervals, nodes))*CPeak*1000; end=None; start=None """

    network = solution.network
    trans_tdc_mask = solution.trans_tdc_mask
    networksteps = np.where(perfect == network.shape[2])[0][0]
    
    Netload = (solution.MLoad - solution.GPV - solution.GWind - solution.GBaseload)[start:end]
    Netload -= flexible
    
    shape2d = intervals, nodes = len(Netload), solution.nodes

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

        Transmissiont=np.zeros((nhvdc, nodes), dtype=np.float64)
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

            Transmissiont = hvdc_even2(Fillt, Surplust, Hcapacity, network, networksteps, Transmissiont)
            
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
            Fillt = (np.maximum(0, Netloadt) #load
                        + np.minimum(Pcapacity, (Scapacity - Storaget_1) / efficiency / resolution) #full charging capacity
                        - Charget) #charge capacity already in use

            Transmissiont = hvdc_even2(Fillt, Surplust, Hcapacity, network, networksteps,
                                 Transmissiont)
            
            Netloadt = Netload[t] - Transmissiont.sum(axis=0)
            Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
            Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)

        Storaget = Storaget_1 - Discharget * resolution + Charget * resolution * efficiency
        
        Discharge[t] = Discharget
        Charge[t] = Charget
        Storage[t] = Storaget
        Import[t] = np.maximum(0, Transmissiont).sum(axis=0)
        Export[t] = np.minimum(0, Transmissiont).sum(axis=0)
        TDC[t] = (Transmissiont*trans_tdc_mask).sum(axis=1)
        
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
def hvdc(Fillt, Surplust, Hcapacity, network, networksteps, Transmissiont):
    for n in range(network.shape[1]):
        if Fillt[n] == 0:
            continue
       
        for leg in range(networksteps):
            donors = network[:, n, perfect[leg]:perfect[leg+1], :]
            donors, donor_lines = donors[0, :, :], donors[1, :, :]
  
            valid_mask = donors[-1] != -1
            if np.prod(~valid_mask):
                break
            donor_lines = donor_lines[:, valid_mask]
            donors = donors[:, valid_mask]
  
            ndonors = valid_mask.sum()
            donors = np.concatenate((n*np.ones((1, ndonors), np.int64), donors))
            d=0
            while Fillt[n] > 0 and d < ndonors:
                #TODO at the moment surplus is taken from first zone, and second isn't 
                #     touched unless necessary. Would be nice to take from all evenly.
                donor_line_cap = np.inf 
                for l in donor_lines[:,d]:
                    # transmission cap is minimum of capacities of lines involved
                    donor_line_cap = min(
                        donor_line_cap, 
                        Hcapacity[l] - np.maximum(0, Transmissiont[l, :]).sum()
                        )
                                          
                _transmission = max(0, min(donor_line_cap, Surplust[donors[-1, d]], Fillt[n]))
               
                for step in range(leg+1): 
                    Transmissiont[donor_lines[step, d], donors[step, d]] += _transmission
                    Transmissiont[donor_lines[step, d], donors[step+1, d]] -= _transmission
  
                Fillt[n] -= _transmission
                Surplust[donors[-1, d]] -= _transmission
                d+=1
               
            if Fillt[n] == 0:
                break
                
    return Transmissiont



@njit
def hvdc_even(Fillt, Surplust, Hcapacity, network, networksteps, Transmissiont):
    """takes energy from neighbours evenly, but takes slightly more than 2x as long to compute"""
    for n in range(network.shape[1]):
        if Fillt[n] <= 0:
            continue
        for leg in range(networksteps):
            donors = network[:, n, perfect[leg]:perfect[leg+1], :]
            donors, donor_lines = donors[0, :, :], donors[1, :, :]
  
            valid_mask = donors[-1] != -1
            if np.prod(~valid_mask):
                break
            donor_lines = donor_lines[:, valid_mask]
            donors = donors[:, valid_mask]
  
            ndonors = valid_mask.sum()
            donors = np.concatenate((n*np.ones((1, ndonors), np.int64), donors))
            
            _transmission = Surplust[donors[-1]]
            if _transmission.sum() == 0:
                continue
            
            CLine= Hcapacity - np.maximum(0, Transmissiont).sum(axis=1)
            
            # reduce _transmission to line capacity
            for line in np.unique(donor_lines):
                di = np.where(donor_lines == line)[1]
                if CLine[line] <= 1e-6:
                    _transmission[di] = 0
                    continue
                _transmission[di] = _transmission[di] / max(1, _transmission[di].sum() / CLine[line])
                    
            # reduce _transmission from donors to not exceed fill requirement
            _transmission /= max(1, _transmission.sum()/Fillt[n])
            
            Fillt[n] -= _transmission.sum()

            Surplust[donors[-1]] -= _transmission
            for step in range(leg+1): 
                for d in range(ndonors):
                    Transmissiont[donor_lines[step, d], donors[step, d]] += _transmission[d]
                    Transmissiont[donor_lines[step, d], donors[step+1, d]] -= _transmission[d]
            
            if Fillt[n] <= 0:
                break
                
    return Transmissiont

@njit
def hvdc_even2(Fillt, Surplust, Hcapacity, network, networksteps, Transmissiont):
    maxconnections = network.shape[-1]

    for leg in range(networksteps):
        Fillmask = Fillt > 1e-6
        donors = network[:, Fillmask, perfect[leg]:perfect[leg+1], :]
        donors, donor_lines = donors[0,:,:,:], donors[1,:,:,:]
    
        valid_mask = donors[:, -1, :] != -1
        
        donors = np.hstack((np.repeat(np.arange(len(Fillt)), maxconnections).reshape(len(Fillt), 1, maxconnections)[Fillmask], donors))
        donors = donors.transpose(1, 0, 2)
        
        _transmission = np.zeros(valid_mask.shape, np.float64)
        valid_mask = (donors[-1, :, :] != -1).flatten()
        
        _transmission.ravel()[valid_mask] = Surplust[donors[-1].ravel()[valid_mask]]

        CLine = Hcapacity - np.maximum(0, Transmissiont).sum(axis=1)

        donor_lines = donor_lines.transpose(1, 0, 2)

        for line in np.unique(donor_lines):
            if line==-1:
                continue
            di = (donor_lines==line).sum(axis=0).astype(np.bool_).ravel()
            if CLine[line] <= 1e-6: 
                _transmission.ravel()[di] = 0
                continue
            _transmission.ravel()[di] /= max(1, _transmission.ravel()[di].sum() / CLine[line])
        
        # divzeromask = Fillt != 0 
        # _transmission[divzeromask] /= np.maximum(1, _transmission.sum(axis=1)[divzeromask]/Fillt[divzeromask])
        # _transmission[~divzeromask] = 0 
        
        _transmission /= np.atleast_2d(np.maximum(1, _transmission.sum(axis=1)/Fillt[Fillmask])).T

        Fillt[Fillmask] -= _transmission.sum(axis=1)
        
        _trans_valid = _transmission.ravel()[valid_mask]
        
        Surplust[donors[-1].ravel()[valid_mask]] -= _trans_valid
        
        for l in range(leg+1):
            for u, ind in enumerate(zip(donor_lines[l].ravel()[valid_mask], donors[l].ravel()[valid_mask])):
                Transmissiont[*ind] += _trans_valid[u]
            for u, ind in enumerate(zip(donor_lines[l].ravel()[valid_mask], donors[l+1].ravel()[valid_mask])):
                Transmissiont[*ind] -= _trans_valid[u]
        
        if Fillt.sum() == 0:
            break
        
    return Transmissiont

        
    
    
    
    
    
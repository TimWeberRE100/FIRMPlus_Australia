# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:53:22 2024

@author: u6942852
"""

import numpy as np
from numba import njit

@njit
def Reliability(solution, flexible, start=None, end=None):
    Nodel_int, PVl_int, Windl_int = solution.Nodel_int, solution.PVl_int, solution.Windl_int
    network = solution.network
    
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
    efficiency, resolution = solution.efficiency, solution.resolution 

    Discharge = np.zeros(shape2d, dtype=np.float64)
    Charge = np.zeros(shape2d, dtype=np.float64)
    Storage = np.zeros(shape2d, dtype=np.float64)
    # Surplus = np.zeros(shape2d, dtype=np.float64)
    Deficit = np.zeros(shape2d, dtype=np.float64)
    Import = np.zeros(shape2d, dtype=np.float64)
    Export = np.zeros(shape2d, dtype=np.float64)
    TDC = np.zeros((intervals, len(network)))

    for t in range(intervals):
        Netloadt = Netload[t]
        Storaget_1 = Storage[t-1,:] if t>0 else 0.5*Scapacity

        Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
        Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
        Deficitt = np.maximum(Netloadt - Discharget ,0)

        Surplust = -1 * np.minimum(0, Netloadt + Charget) 
       
        Importt = np.zeros((len(network), nodes), dtype=np.float64)
        Exportt = np.zeros((len(network), nodes), dtype=np.float64) 
        
        if Deficitt.sum() > 1e-6:
            # Fill deficits with transmission without drawing down from battery reserves
            fill_req = np.maximum(Netloadt - Discharget, 0)
            fill_req, Surplust, Importt, Exportt, DischargeSurplust = hvdc_control(
                fill_req, Surplust, Importt, Exportt, Netloadt, np.zeros(nodes, np.float64), Hcapacity, Nodel_int, network)
            
            Netloadt = Netload[t] - Importt.sum(axis=0) + Exportt.sum(axis=0)
            Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
            Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
            Deficitt = np.maximum(Netloadt - Discharget ,0)
    
        if Deficitt.sum() > 1e-6: 
            # Fill deficits with transmission by drawing down from battery reserves
            fill_req = np.maximum(Netloadt - Discharget, 0)
            Surplust = -1 * np.minimum(0, Netloadt + Charget) 
            DischargeSurplust = np.minimum(Pcapacity, Storaget_1 / resolution)
            
            fill_req, Surplust, Importt, Exportt, DischargeSurplust = hvdc_control(
                fill_req, np.zeros(nodes, np.float64), Importt, Exportt, Netloadt, DischargeSurplust, Hcapacity, Nodel_int, network)
            
            Netloadt = Netload[t] - Importt.sum(axis=0) + Exportt.sum(axis=0)
            Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
            Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)

        # =============================================================================
        # To Do: If deficit Go back in time and discharge batteries 
        # This may make the time a fair bit longer
        # =============================================================================

        if Surplust.sum() > 1e-6:
            # Distribute surplus energy with transmission to areas with spare charging capacity
            fill_req = (np.maximum(0, Netloadt) #load
                        + np.minimum(Pcapacity, (Scapacity - Storaget_1) / efficiency / resolution) #real charging capacity
                        - Charget) #charge capacity in use
            fill_req, Surplust, Importt, Exportt, DischargeSurplust = hvdc_control(
                fill_req, Surplust, Importt, Exportt, Netloadt, np.zeros(nodes, np.float64), Hcapacity, Nodel_int, network)
            
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
def hvdc_control(fill_req, Surplust, Importt, Exportt, Netloadt, DischargeSurplust, Hcapacity, Nodel_int, network):
    """Note, one of DischargeSurplust and Surplust should be np.zeros"""
    for n in range(len(Nodel_int)):
        if fill_req[n] == 0:
            continue
        leg1_h = np.where((network == Nodel_int[n]).sum(axis=1) == 1)[0]
        leg1_net = np.unique(network[leg1_h, :])
        leg1_net = np.array([a for a, b in enumerate(Nodel_int) if b in leg1_net and b != Nodel_int[n]])
        
        if len(leg1_net) == 0: 
            continue

        fill_req, Surplust, Importt, Exportt, DischargeSurplust = hvdc_leg1(
            leg1_net, leg1_h, n, fill_req, Surplust, 
            Importt, Exportt, Netloadt, DischargeSurplust, Hcapacity)
    
        if fill_req[n] == 0: 
            continue
        
        for o_net, o_h in zip(leg1_net, leg1_h):
            leg2_h = np.where((network == Nodel_int[o_net]).sum(axis=1) * 
                              ((network != Nodel_int[n]).sum(axis=1)==2))[0]
            leg2_net = np.unique(network[leg2_h, :])
            leg2_net = np.array([a for a, b in enumerate(Nodel_int) 
                                 if b in leg2_net 
                                 and b not in (Nodel_int[o_net], Nodel_int[n])])
            
            if len(leg2_net)==0:
                continue
            fill_req, Surplust, Importt, Exportt, DischargeSurplust = hvdc_leg2(
                o_net, o_h, leg2_net, leg2_h, n, fill_req, Surplust, 
                Importt, Exportt, Netloadt, DischargeSurplust, Hcapacity)

            if fill_req[n] == 0: 
                break
        if fill_req[n] == 0: 
            continue
        
        for o_net, o_h in zip(leg1_net, leg1_h):
            leg2_h = np.where((network == Nodel_int[o_net]).sum(axis=1) * 
                              ((network != Nodel_int[n]).sum(axis=1)==2))[0]
            leg2_net = np.unique(network[leg2_h, :])
            leg2_net = np.array([a for a, b in enumerate(Nodel_int) 
                                 if b in leg2_net 
                                 and b not in (Nodel_int[o_net], Nodel_int[n])])
            if len(leg2_net)==0:
                continue
            for p_net, p_h in zip(leg2_net, leg2_h):
                leg3_h = np.where((network == Nodel_int[p_net]).sum(axis=1) * 
                                  ((network != Nodel_int[o_net]).sum(axis=1)==2) * 
                                  ((network != Nodel_int[n]).sum(axis=1)==2))[0]
                leg3_net = np.unique(network[leg3_h, :])
                leg3_net = np.array([a for a, b in enumerate(Nodel_int) 
                                     if b in leg3_net 
                                     and b not in (Nodel_int[p_net], Nodel_int[o_net], Nodel_int[n])])
                if len(leg3_net)==0:
                    continue
                fill_req, Surplust, Importt, Exportt, DischargeSurplust = hvdc_leg3(
                    p_net, p_h, o_net, o_h, leg3_net, leg3_h, n, fill_req, Surplust, 
                    Importt, Exportt, Netloadt, DischargeSurplust, Hcapacity)
                if fill_req[n] == 0: 
                    break
            if fill_req[n] == 0: 
                break
        if fill_req[n] == 0: 
            continue
        
        for o_net, o_h in zip(leg1_net, leg1_h):
            leg2_h = np.where((network == Nodel_int[o_net]).sum(axis=1) * 
                              ((network != Nodel_int[n]).sum(axis=1)==2))[0]
            leg2_net = np.unique(network[leg2_h, :])
            leg2_net = np.array([a for a, b in enumerate(Nodel_int) 
                                 if b in leg2_net 
                                 and b not in (Nodel_int[o_net], Nodel_int[n])])
            if len(leg2_net) == 0:
                continue
            for p_net, p_h in zip(leg2_net, leg2_h):
                leg3_h = np.where((network == Nodel_int[p_net]).sum(axis=1) * 
                                  ((network != Nodel_int[o_net]).sum(axis=1)==2) * 
                                  ((network != Nodel_int[n]).sum(axis=1)==2))[0]
                leg3_net = np.unique(network[leg3_h, :])
                leg3_net = np.array([a for a, b in enumerate(Nodel_int) 
                                     if b in leg3_net 
                                     and b not in (Nodel_int[p_net], Nodel_int[o_net], Nodel_int[n])])
                if len(leg3_net)==0:
                    continue
                for q_net, q_h in zip(leg3_net, leg3_h):
                    leg4_h = np.where((network == Nodel_int[q_net]).sum(axis=1) * 
                                      ((network != Nodel_int[p_net]).sum(axis=1)==2) * 
                                      ((network != Nodel_int[o_net]).sum(axis=1)==2) * 
                                      ((network != Nodel_int[n]).sum(axis=1)==2))[0]
                    leg4_net = np.unique(network[leg4_h, :])
                    leg4_net = np.array([a for a, b in enumerate(Nodel_int) 
                                         if b in leg4_net 
                                         and b not in (Nodel_int[q_net], Nodel_int[p_net], Nodel_int[o_net], Nodel_int[n])])
                    if len(leg4_net)==0:
                        continue
                    fill_req, Surplust, Importt, Exportt, DischargeSurplust = hvdc_leg4(
                        q_net, q_h, p_net, p_h, o_net, o_h, leg4_net, leg4_h, n, fill_req, Surplust, 
                        Importt, Exportt, Netloadt, DischargeSurplust, Hcapacity)
                if fill_req[n] == 0: 
                    break
            if fill_req[n] == 0: 
                break
        if fill_req[n] == 0: 
            continue
        
    return fill_req, Surplust, Importt, Exportt, DischargeSurplust

@njit 
def hvdc_leg1(leg1_net, leg1_h, n, fill_req, Surplust, Importt, Exportt, 
              Netloadt, DischargeSurplust, Hcapacity):
    """Note, one of DischargeSurplust and Surplust should be np.zeros"""
    _rec = 0
    while fill_req[n] > 0 and _rec < len(leg1_net):
        _available = np.maximum(
            0, 
            np.minimum(
                np.minimum(
                    np.maximum(
                        Surplust[leg1_net], 
                        DischargeSurplust[leg1_net]), #surplus 
                    fill_req[n]), # energy need
                Hcapacity[leg1_h] - np.atleast_2d(Exportt[leg1_h, :]).sum(axis=1))) #transmission
        _amask = _available > 0 
        transmission = np.minimum(
            _available, 
            fill_req[n]/ _amask.sum() if _amask.sum()>=1 else 0)
    
        # numba does not handle multidimensional indexing
        for l, trans in enumerate(transmission):
            Exportt[leg1_h[l], leg1_net[l]] += trans
            Importt[leg1_h[l], n] += trans
            Surplust[leg1_net[l]] -= trans
            DischargeSurplust[leg1_net[l]] -= trans
        _rec += 1
        fill_req[n] -= transmission.sum()

    return fill_req, Surplust, Importt, Exportt, DischargeSurplust

@njit
def hvdc_leg2(o_net, o_h, leg2_net, leg2_h,  n, fill_req, Surplust, Importt, Exportt, 
              Netloadt, DischargeSurplust, Hcapacity):
    """Note, one of DischargeSurplust and Surplust should be np.zeros"""
    _rec = 0
    while fill_req[n] > 0 and _rec < len(leg2_net):
        _available = np.maximum(
            np.minimum(
                np.minimum(
                    np.minimum(
                        np.maximum(
                            Surplust[leg2_net], 
                            DischargeSurplust[leg2_net]), #surplus 
                        fill_req[n]), # energy need
                    Hcapacity[leg2_h] - np.atleast_2d(Exportt[leg2_h, :]).sum(axis=1)),  #transmission leg 1
                Hcapacity[o_h] - np.atleast_2d(Exportt[o_h, :]).sum(axis=1)), #transmission leg 2
            0) 
        _amask = _available > 0 
        transmission = np.minimum(
            _available, 
            fill_req[n]/_amask.sum() if _amask.sum()>=1 else 0)
        for l, trans in enumerate(transmission):
            Exportt[leg2_h[l], leg2_net[l]] += trans #transmission leg 1
            Importt[leg2_h[l], o_net] += trans #transmission leg 1
            Exportt[o_h, o_net] += trans #transmission leg 2
            Importt[o_h, n] += trans #transmission leg 2
            Surplust[leg2_net[l]] -= trans
            DischargeSurplust[leg2_net[l]] -= trans

        _rec+=1
        fill_req[n] -= transmission.sum()
    return fill_req, Surplust, Importt, Exportt, DischargeSurplust
            
@njit
def hvdc_leg3(p_net, p_h, o_net, o_h, leg3_net, leg3_h, n, fill_req, Surplust, Importt, Exportt, 
              Netloadt, DischargeSurplust, Hcapacity):
    """Note, one of DischargeSurplust and Surplust should be np.zeros"""
    _rec = 0
    while fill_req[n] > 0 and _rec < len(leg3_net):
        _available = np.maximum(
            np.minimum(
                np.minimum(
                    np.minimum(
                        np.minimum(
                            np.maximum(
                                Surplust[leg3_net], 
                                DischargeSurplust[leg3_net]), #surplus 
                            fill_req[n]), # energy need
                        Hcapacity[leg3_h] - np.atleast_2d(Exportt[leg3_h, :]).sum(axis=1)),  #transmission leg 1
                    Hcapacity[p_h] - np.atleast_2d(Exportt[p_h, :]).sum(axis=1)), #transmission leg 2
                Hcapacity[o_h] - np.atleast_2d(Exportt[o_h, :]).sum(axis=1)),#transmission leg 3
            0) 
        _amask = _available > 0 
        transmission = np.minimum(
            _available, 
            fill_req[n]/_amask.sum() if _amask.sum()>=1 else 0)
        for l, trans in enumerate(transmission):
            Exportt[leg3_h[l], leg3_net[l]] += trans #transmission leg 1
            Importt[leg3_h[l], p_net] += trans #transmission leg 1
            Exportt[p_h, p_net] += trans #transmission leg 2
            Importt[p_h, o_net] += trans #transmission leg 2
            Exportt[o_h, o_net] += trans #transmission leg 3
            Importt[o_h, n] += trans #transmission leg 3
            Surplust[leg3_net] -= trans
            DischargeSurplust[leg3_net] -= trans
        _rec+=1
        fill_req[n] -= transmission.sum()
    return fill_req, Surplust, Importt, Exportt, DischargeSurplust
            
            
@njit
def hvdc_leg4(q_net, q_h, p_net, p_h, o_net, o_h, leg4_net, leg4_h, n, fill_req, Surplust, Importt, Exportt, 
              Netloadt, DischargeSurplust, Hcapacity):
    """Note, one of DischargeSurplust and Surplust should be np.zeros"""
    _rec = 0
    while fill_req[n] > 0 and _rec < len(leg4_net):
        _available = np.maximum(
            np.minimum(
                np.minimum(
                    np.minimum(
                        np.minimum(
                            np.minimum(
                                np.maximum(
                                    Surplust[leg4_net], 
                                    DischargeSurplust[leg4_net]), #surplus 
                                fill_req[n]), # energy need
                            Hcapacity[leg4_h] - np.atleast_2d(Exportt[leg4_h, :]).sum(axis=1)),#transmission leg 1
                        Hcapacity[q_h] - np.atleast_2d(Exportt[q_h, :]).sum(axis=1)),#transmission leg 2
                    Hcapacity[p_h] - np.atleast_2d(Exportt[p_h, :]).sum(axis=1)),#transmission leg 3
                Hcapacity[o_h] - np.atleast_2d(Exportt[o_h, :]).sum(axis=1)),#transmission leg 4
            0) 
        _amask = _available > 0 
        transmission = np.minimum(
            _available, 
            fill_req[n]/_amask.sum() if _amask.sum()>=1 else 0)
        for l, trans in enumerate(transmission):
            Exportt[leg4_h[l], leg4_net[l]] += trans #transmission leg 1
            Importt[leg4_h[l], q_net] += trans #transmission leg 1
            Exportt[q_h, q_net] += trans #transmission leg 2
            Importt[q_h, p_net] += trans #transmission leg 2
            Exportt[p_h, p_net] += trans #transmission leg 3
            Importt[p_h, o_net] += trans #transmission leg 3
            Exportt[o_h, o_net] += trans #transmission leg 3
            Importt[o_h, n] += trans #transmission leg 3
            Surplust[leg4_net] -= trans 
            DischargeSurplust[leg4_net] -= trans 
        _rec+=1
        fill_req[n] -= transmission.sum()
    return fill_req, Surplust, Importt, Exportt, DischargeSurplust
    
    
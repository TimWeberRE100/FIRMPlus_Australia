# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 07:51:51 2024

@author: u6942852
"""


import numpy as np
from numba import njit  # type: ignore

from firm.Utils import cclock, array_min, array_max_2d_axis1, array_sum_2d_axis0, zero_safe_division # type: ignore


@njit
def get_primary_donors(solution, n):
    cache_result = solution.cache_primary_donors.get(n, None)
    if cache_result is not None:
        return cache_result

    result = solution.network[:, n, 0, :]
    result = result[:, result[0] != -1]
    solution.cache_primary_donors[n] = result
    return result

@njit
def get_nthary_donors(solution, n, leg):
    if leg == 1: 
        cache = solution.cache_secondary_donors
    elif leg == 2:
        cache = solution.cache_tertiary_donors
    elif leg == 3:
        cache = solution.cache_quaternary_donors
    
    cache_result = cache.get(n, None)
    if cache_result is not None:
        return cache_result
    
    result = solution.network[:, n, solution.triangulars[leg] : solution.triangulars[leg + 1], :]
    result = result[:, :, result[0, 0, :] != -1]
    cache[n] = result
    return result
    


@njit
def Interconnection(solution, Fillt, Surplust, Importt, Exportt):
    # The primary connections are simpler (and faster) to model than the general
    #   nthary connection
    # Since many if not most calls of this function only require primary transmission
    #   I have split it out from general nthary transmission to improve speed
    if solution.profiling:
        time_start = cclock()
    _transmission = np.zeros(solution.nodes, np.float64)
    leg = 0
    # loop through nodes with deficits
    for n in range(solution.nodes):
        if Fillt[n] < 1e-6:
            continue
        # appropriate slice of network array
        # pdonors is equivalent to donors later on but has different ndim so needs to
        #   be a different variable name for static typing
        pdonors, pdonor_lines = get_primary_donors(solution, n)
        _usage = 0.0 # badly named by avoids creating more variables
        for d in pdonors: 
            _usage += Surplust[d]

        if _usage < 1e-6:
            # continue if no surplus to be traded
            continue

        for d, l in zip(pdonors, pdonor_lines):
            _usage = 0.0
            for m in range(solution.nodes):
                _usage += Importt[m, l]
            # maximum exportable
            _transmission[d] = min(
                Surplust[d],  # power resource constraint
                solution.CHVI[l] - _usage, # line capacity constraint
            )  

        # scale down to fill requirement
        _usage = 0.0
        for m in range(solution.nodes):
            _usage += _transmission[m] 
        if _usage > Fillt[n]:
            _scale = Fillt[n] / _usage
            _transmission *= _scale
            _usage *= _scale
        if _usage < 1e-6:
            continue

        # for d, l in zip(pdonors, pdonor_lines):  #  print(d,l)
        for i in range(len(pdonors)):
            # record transmission
            Importt[n, pdonor_lines[i]] += _transmission[pdonors[i]]
            Exportt[pdonors[i], pdonor_lines[i]] -= _transmission[pdonors[i]]
            # adjust deficit/surpluses
            Surplust[pdonors[i]] -= _transmission[pdonors[i]]
            _transmission[pdonors[i]] = 0
 
        Fillt[n] -= _usage
    
    if solution.profiling:
        solution.calls_interconnection0 +=1
        solution.time_interconnection0 += cclock() - time_start

    # Continue with nthary transmission
    # Note: This code block works for primary transmission too, but is slower
    if (Fillt.sum() > 1e-6) and (Surplust.sum() > 1e-6):
            _import = np.zeros(Importt.shape, np.float64)
            _capacity = np.zeros(solution.nhvi, np.float64)
            # loop through secondary, tertiary, ..., nthary connections
            for leg in range(1, solution.networksteps):
                if solution.profiling:
                    time_start = cclock()

                # loop through nodes with deficits
                for n in range(solution.nodes):
                    if Fillt[n] < 1e-6:
                        continue
                    
                    donors, donor_lines = get_nthary_donors(solution, n, leg)

                    if donors.shape[1] == 0:
                        break  # break if no valid donors
                        
                    _usage = 0.0 # badly named variable but avoids extra variables
                    for d in donors[-1]:
                        _usage += Surplust[d]

                    if _usage < 1e-6:
                        continue

                    _capacity[:] = solution.CHVI - array_sum_2d_axis0(Importt)
                    for d, dl in zip(donors[-1], donor_lines.T): # print(d,dl)
                        # power use of each line, clipped to maximum capacity of lowest leg
                        _import[d, dl] = min(array_min(_capacity[dl]), Surplust[d])
                    
                    for l in range(solution.nhvi):
                        # total usage of the line across all import paths
                        _usage=0.0
                        for m in range(solution.nodes):
                            _usage += _import[m, l]
                        # if usage exceeds capacity
                        if _usage > _capacity[l]:
                            # unclear why this raises zero division error from time to time
                            _scale = zero_safe_division(_capacity[l], _usage)
                            for m in range(solution.nodes):
                                # clip all legs
                                if _import[m, l] > 1e-6:
                                    for o in range(solution.nhvi):
                                        _import[m, o] *= _scale
                        
                    # intermediate calculation array
                    _transmission = array_max_2d_axis1(_import)
                    
                    # scale down to fill requirement
                    _usage = 0.0
                    for m in range(solution.nodes):
                        _usage += _transmission[m] 
                    if _usage > Fillt[n]:
                        _scale = Fillt[n] / _usage
                        _transmission *= _scale
                        _usage *= _scale
                    if _usage < 1e-6:
                        continue

                    for nd, d, dl in zip(range(donors.shape[1]), donors[-1], donor_lines.T): # print(nd, d, dl)
                        Importt[n, dl[0]] += _transmission[d]
                        Exportt[donors[0, nd], dl[0]] -= _transmission[d]
                        for step in range(leg):
                            Importt[donors[step, nd], dl[step+1]] += _transmission[d]
                            Exportt[donors[step+1, nd], dl[step+1]] -= _transmission[d]

                    # Adjust fill and surplus
                    Fillt[n] -= _usage
                    Surplust -= _transmission
                    
                    _import[:] = 0.0
                    _capacity[:] = 0.0
                    
                    if (Surplust.sum() < 1e-6) or (Fillt.sum() < 1e-6):
                        break
                    # if not (Surplust > 1e-6).any():
                    #     break
                    # if not (Fillt > 1e-6).any():
                    #     break

                if solution.profiling:
                    if leg == 1:
                        solution.calls_interconnection1 +=1
                        solution.time_interconnection1 += cclock() - time_start
                    elif leg == 2:
                        solution.calls_interconnection1 +=1
                        solution.time_interconnection1 += cclock() - time_start
                    elif leg == 3:
                        solution.calls_interconnection3 +=1
                        solution.time_interconnection3 += cclock() - time_start

                if (Surplust.sum() < 1e-6) or (Fillt.sum() < 1e-6):
                    break
                # if not (Surplust > 1e-6).any():
                #     break
                # if not (Fillt > 1e-6).any():
                #     break

    return Importt, Exportt


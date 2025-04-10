# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 07:51:51 2024

@author: u6942852
"""


import numpy as np
from numba import njit  # type: ignore

from firm.Utils import cclock  # type: ignore

triangulars = np.array([0, 1, 3, 6, 10, 15, 21])


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
    
    result = solution.network[:, n, triangulars[leg] : triangulars[leg + 1], :]
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
    _transmission = np.zeros_like(Fillt)
    leg = 0
    # loop through nodes with deficits
    for n, f in enumerate(Fillt):
        if f < 1e-6:
            continue
        # appropriate slice of network array
        # pdonors is equivalent to donors later on but has different ndim so needs to
        #   be a different variable name for static typing
        # pdonors = solution.network[:, n, 0, :]
        # valid_mask = pdonors[0] != -1
        # # donor nodes and donor lines
        # pdonors, pdonor_lines = pdonors[0, valid_mask], pdonors[1, valid_mask]

        pdonors, pdonor_lines = get_primary_donors(solution, n)
        if Surplust[pdonors].sum() < 1e-6:
            # continue if no surplus to be traded
            continue

        # maximum exportable
        _transmission[pdonors] = np.minimum(
            Surplust[pdonors],  # power resource constraint
            solution.CHVI[pdonor_lines] - Importt[pdonor_lines, :].sum(axis=1), # line capacity constraint
        )  
        # scale down to fill requirement
        _transmission /= max(1, _transmission.sum() / Fillt[n])

        if _transmission.sum() < 1e-6:
            continue

        for d, l in zip(pdonors, pdonor_lines):  #  print(d,l)
            # record transmission
            Importt[l, n] += _transmission[d]
            Exportt[l, d] -= _transmission[d]

        # adjust deficit
        Fillt[n] -= _transmission.sum()
        # adjust surpluses
        Surplust -= _transmission
        _transmission[:] = 0

    # Continue with nthary transmission
    # Note: This code block works for primary transmission too, but is slower
    if Surplust.sum() > 1e-6 and Fillt.sum() > 1e-6:
        _import = -1 * np.ones(Importt.shape, np.float64)
        _hostingcapacity = np.zeros(solution.nhvi, np.float64)
        # loop through secondary, tertiary, ..., nthary connections
        for leg in range(1, solution.networksteps):
            # loop through nodes with deficits
            for n, f in enumerate(Fillt):
                if f < 1e-6:
                    continue
                donors = solution.network[:, n, triangulars[leg] : triangulars[leg + 1], :]
                donors, donor_lines = donors[0, :, :], donors[1, :, :]

                valid_mask = donors[-1] != -1
                
                pdonors, pdonor_lines = get_nthary_donors(solution, n, leg)

                if pdonors.shape[1] == 0:
                    break  # break if no valid donors
                donors, donor_lines = donors[:, valid_mask], donor_lines[:, valid_mask]
                # donors[-1] is where power comes from. donors[:-1] is are nodes it travels through
                if Surplust[donors[-1]].sum() < 1e-6:
                    continue

                for d, dl in zip(donors[-1], donor_lines.T):  # print(d,dl)
                    # power use of each line
                    _import[dl, d] = Surplust[d]
                _hostingcapacity[:] = solution.CHVI - Importt.sum(axis=1)
                
                for i in range(solution.nhvi):
                    if _hostingcapacity[i] > 0.0:
                        row_sum =  _import[i, :].sum()
                        if row_sum > _hostingcapacity[i]:
                            scaling = _hostingcapacity[i] / row_sum
                            _import[i, :] *= scaling
                    else:
                        _import[i, :] = 0.0
                        
                # intermediate calculation array
                _transmission = _import.sum(axis=0)
                # transmission is the least amount that any one line in the chain can host
                for i in range(solution.nhvi):  # print(_row)
                    for j in range(solution.nodes):
                        if _import[i, j] == -1 :
                            continue
                        else:
                            if _import[i, j] < _transmission[j]:
                                _transmission[j] = _import[i, j]
                            # _transmission[j] = min(_transmission[j], _import[i, j]) 
               
                # remove invalid values
                _transmission = np.maximum(0, _transmission)
                # scale down to fill requirement
                _transmission /= max(1, _transmission.sum() / Fillt[n])

                # add receiver to start of donors
                donors = np.concatenate((np.full((1, pdonors.shape[1]), n), donors))

                for nd, d, dl in zip(range(pdonors.shape[1]), donors[-1], donor_lines.T):
                    for step, l in enumerate(dl):
                        Importt[l, donors[step, nd]] += _transmission[d]
                        Exportt[l, donors[step + 1, nd]] -= _transmission[d]

                # Adjust fill and surplus
                Fillt[n] -= _transmission.sum()
                Surplust -= _transmission
                
                _import[:] = -1.0
                _hostingcapacity[:] = 0
                
                if Surplust.sum() < 1e-6 or Fillt.sum() < 1e-6:
                    break

            if Surplust.sum() < 1e-6 or Fillt.sum() < 1e-6:
                break
    if solution.profiling:
        if leg == 0:
            solution.time_interconnection0 += cclock() - time_start
            solution.calls_interconnection0 +=1
        elif leg == 1:
            solution.time_interconnection1 += cclock() - time_start
            solution.calls_interconnection1 +=1
        elif leg == 2:
            solution.time_interconnection1 += cclock() - time_start
            solution.calls_interconnection1 +=1
        elif leg == 3:
            solution.time_interconnection3 += cclock() - time_start
            solution.calls_interconnection3 +=1
    return Importt, Exportt

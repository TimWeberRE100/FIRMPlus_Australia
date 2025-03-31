# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 07:51:51 2024

@author: u6942852
"""

import numpy as np
from numba import njit

triangulars = np.array([0,1,3,6,10,15,21])

@njit
def Interconnection(solution, Fillt, Surplust, Importt, Exportt):
    # The primary connections are simpler (and faster) to model than the general
    #   nthary connection
    # Since many if not most calls of this function only require primary transmission
    #   I have split it out from general nthary transmission to improve speed
    
    # loop through nodes with deficits 
    for n, f in enumerate(Fillt):
        if f < 1e-6:
            continue
        # appropriate slice of network array    
        # pdonors is equivalent to donors later on but has different ndim so needs to 
        #   be a different variable name for static typing
        pdonors = solution.network[:, n, 0, :] 
        valid_mask = pdonors[0] != -1
        # donor nodes and donor lines 
        pdonors, pdonor_lines = pdonors[0, valid_mask], pdonors[1, valid_mask]
        
        if Surplust[pdonors].sum() <= 1e-6:
            # continue if no surplus to be traded
            continue
  
        # intermediate calculation array
        _transmission = np.zeros_like(Fillt)
        
        # maximum exportable
        _transmission[pdonors] = np.minimum(
            Surplust[pdonors], # power resource constraint 
            solution.CHVI[pdonor_lines]-Importt[pdonor_lines,:].sum(axis=1)) # line capacity constraint
        # scale down to fill requirement
        _transmission /= max(1, _transmission.sum()/Fillt[n])
        
        for d, l in zip(pdonors, pdonor_lines):#  print(d,l)
            # record transmission
            Importt[l, n] += _transmission[d]
            Exportt[l, d] -= _transmission[d]
        
        # adjust deficit 
        Fillt[n] -= _transmission.sum()
        # adjust surpluses
        Surplust -= _transmission                

    # Continue with nthary transmission 
    # Note: This code block works for primary transmission too, but is slower
    if Surplust.sum() > 1e-6 and Fillt.sum() > 1e-6:
        # loop through secondary, tertiary, ..., nthary connections
        for leg in range(1, solution.networksteps):
            # loop through nodes with deficits
            for n, f in enumerate(Fillt):
                if f < 1e-6:
                    continue
                donors = solution.network[:, n, triangulars[leg]:triangulars[leg+1], :]
                donors, donor_lines = donors[0, :, :], donors[1, :, :]
      
                valid_mask = donors[-1] != -1
                ndonors = valid_mask.sum()
                if ndonors==0:
                    break # break if no valid donors
                donors, donor_lines = donors[:, valid_mask], donor_lines[:, valid_mask]
                # donors[-1] is where power comes from. donors[:-1] is are nodes it travels through
                if Surplust[donors[-1]].sum() <= 1e-6:
                    continue
                # add receiver to start of donors
                donors = np.concatenate((np.full((1, ndonors), n), donors))
                
                # intermediate calculation array
                _import = np.zeros_like(Importt)
                for d, dl in zip(donors[-1], donor_lines.T): # print(d,dl)
                    # power use of each line
                    _import[dl, d] = Surplust[d]
                
                # remaining hosting capacity of each line
                hostingcapacity = (solution.CHVI-Importt.sum(axis=1))
                zmask = hostingcapacity > 0
                # scale down transmission according to hosting capacity
                _import[zmask] /= np.atleast_2d(np.maximum(1, _import.sum(axis=1)/hostingcapacity)).T[zmask]
                # invalid values in zmask
                _import[~zmask]*=-1
                # intermediate calculation array
                _transmission = _import.sum(axis=0)
                # transmission is the least amount that any one line in the chain can host
                for _row in _import:
                    zmask = _row!=0
                    _transmission[zmask] = np.minimum(_row, _transmission)[zmask]
                # remove invalid values
                _transmission=np.maximum(0, _transmission)
                # scale down to fill requirement
                _transmission /= max(1, _transmission.sum()/Fillt[n])
                
                # add all this info to our operations log
                for nd, d, dl in zip(range(ndonors), donors[-1], donor_lines.T):
                    for step, l in enumerate(dl): 
                        Importt[l, donors[step, nd]] += _transmission[d]
                        Exportt[l, donors[step+1, nd]] -= _transmission[d]
                # Adjust fill and surplus
                Fillt[n] -= _transmission.sum()
                Surplust -= _transmission                
                
                if Surplust.sum() <= 1e-6 or Fillt.sum() <= 1e-6:
                    break
                
            if Surplust.sum() <= 1e-6 or Fillt.sum() <= 1e-6:
                break
        
    return Importt, Exportt
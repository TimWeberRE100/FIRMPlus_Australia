# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 07:51:51 2024

@author: u6942852
"""

import numpy as np
from numba import njit

triangulars = np.array([0,1,3,6,10,15,21])

@njit
def hvdc(solution, Fillt, Surplust, Importt, Exportt):
    # The primary connections are simpler (and faster) to model than the general
    #   nthary connection
    # Since many if not most calls of this function only require primary transmission
    #   I have split it out from general nthary transmission to improve speed
    for n in np.where(Fillt>0)[0]:
        pdonors = solution.network[:, n, 0, :]
        valid_mask = pdonors[0] != -1
        pdonors, pdonor_lines = pdonors[0, valid_mask], pdonors[1, valid_mask]
  
        if Surplust[pdonors].sum() == 0:
            continue
  
        _transmission = np.zeros_like(Fillt)
        _transmission[pdonors] = Surplust[pdonors]
        _transmission[pdonors] = np.minimum(_transmission[pdonors], solution.CHVDC[pdonor_lines]-Importt[pdonor_lines,:].sum(axis=1))
        
        _transmission /= max(1, _transmission.sum()/Fillt[n])
        
        for d, l in zip(pdonors, pdonor_lines):#  print(d,l)
            Importt[l, n] += _transmission[d]
            Exportt[l, d] -= _transmission[d]
            
        Fillt[n] -= _transmission.sum()
        Surplust -= _transmission                

    # Continue with nthary transmission 
    # Note: This code block works for primary transmission too, but is slower
    if Surplust.sum() > 0 and Fillt.sum() > 0:
        for leg in range(1, solution.networksteps):
            for n in np.where(Fillt>0)[0]:
                donors = solution.network[:, n, triangulars[leg]:triangulars[leg+1], :]
                donors, donor_lines = donors[0, :, :], donors[1, :, :]
      
                valid_mask = donors[-1] != -1
                if np.prod(~valid_mask):
                    break
                donor_lines = donor_lines[:, valid_mask]
                donors = donors[:, valid_mask]
                if Surplust[donors[-1]].sum() == 0:
                    continue
      
                ndonors = valid_mask.sum()
                donors = np.concatenate((n*np.ones((1, ndonors), np.int64), donors))
                
                _import = np.zeros_like(Importt)
                for d, dl in zip(donors[-1], donor_lines.T): #print(d,dl)
                    _import[dl, d] = Surplust[d]
                
                hostingcapacity = (solution.CHVDC-Importt.sum(axis=1))
                zmask = hostingcapacity > 0
                _import[zmask] /= np.atleast_2d(np.maximum(1, _import.sum(axis=1)/hostingcapacity)).T[zmask]
                _import[~zmask]*=-1
                _transmission = _import.sum(axis=0)
                for _row in _import:
                    zmask = _row!=0
                    _transmission[zmask] = np.minimum(_row, _transmission)[zmask]
                _transmission=np.maximum(0, _transmission)
                _transmission /= max(1, _transmission.sum()/Fillt[n])
                
                for nd, d, dl in zip(range(ndonors), donors[-1], donor_lines.T):
                    for step, l in enumerate(dl): 
                        Importt[l, donors[step, nd]] += _transmission[d]
                        Exportt[l, donors[step+1, nd]] -= _transmission[d]
                Fillt[n] -= _transmission.sum()
                Surplust -= _transmission                
                
                if Surplust.sum() == 0 or Fillt.sum() == 0:
                    break
                
            if Surplust.sum() == 0 or Fillt.sum() == 0:
                break
        
    return Importt, Exportt
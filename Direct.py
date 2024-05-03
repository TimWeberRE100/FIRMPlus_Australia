# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:27:19 2024

@author: u6942852
"""


import numpy as np
import datetime as dt
from numba import jit, objmode
from csv import writer

from Input import *
from DirectAlgorithm import Direct



@njit
def gen_inds(npop, vsize=args.vp):
    range_gen = range(npop//vsize + 1) if npop%vsize != 0 else range(npop//vsize)
    indcs = [np.arange(n*vsize, min((n+1)*vsize, npop), dtype=np.int64) for n in range_gen]
    
    inds = []
    for i in range(len(indcs)):
        inds.append(indcs[i])
    return inds


# @jit(nopython=False)
def Obj(x, callback=False):
    
    S = Solution(x)
    S._evaluate()
    result = S.Lcoe + S.Penalties
    
    if callback is True:
        printout = np.array([result] + list(x))
        csvfile = open('Results/History{}.csv'.format(scenario), 'a', newline='')
        writer(csvfile).writerow(printout)
        csvfile.close()
    return result
    
    
@jit(nopython=False)
def Vobj(x, callback=False, maxvectorwidth=args.vp, gen=0, cuts=0):
    #TODO 
    # Better way of gen and cuts
    results = np.empty(len(x.T), dtype=np.float64)
    
    inds = gen_inds(len(x.T), maxvectorwidth)
    for ind in inds:
        S = VSolution(x[:, ind])
        S._evaluate()
        results[ind] = S.Lcoe + S.Penalties
    
    if callback is True:
        gen = gen*np.ones((len(x.T), 1), dtype=np.int64)
        cuts = cuts*np.ones((len(x.T), 1), dtype=np.int64)
        
        printout = np.concatenate((results.reshape(-1, 1), gen, cuts, x.T), axis = 1)
        with objmode():
            csvfile = open('Results/History{}.csv'.format(scenario), 'a', newline='')
            writer(csvfile).writerows(printout)
            csvfile.close()
    return results
    
def Callback_1(h):
    with open('Results/History{}.csv'.format(scenario), 'a', newline='') as csvfile:
        writer(csvfile).writerow([h.f] + list(h.centre))
        
def Init_callback():
    with open('Results/History{}.csv'.format(scenario), 'w', newline='') as csvfile:
        writer(csvfile)

if __name__ == '__main__':
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)
    
    z = (pzones+wzones+nodes)
    ultralow_res = np.array([1.0]*z + [100.0]) # 1 GW, 100 GWh
    low_res = np.array([0.1]*z + [10.0]) # 100 MW, 10 GWh
    medium_res = np.array([0.01]*z + [1.0]) # 10 MW, 1 GWh
    high_res = np.array([0.001]*z + [0.1]) # 1 MW, 100 MWh
    ultrahigh_res = np.array([0.000_1]*z + [0.01]) # 0.1 MW, 10 MWh
    polishing = np.array([0.000_001]*z + [0.000_1]) # 1 kW, 100 kWh
    
    res = [ultralow_res, low_res, medium_res, high_res, ultrahigh_res, polishing]
    
    result = Direct(
        func=Vobj if args.vec else Obj, 
        bounds=(lb, ub),
        f_args=(args.cb==2, args.vp) if args.vec else (args.cb==2,),
        # maxiter=args.i, 
        callback= Callback_1 if args.cb==1 else None, 
        vectorizable=bool(args.vec),
        workers=args.w,
        population=args.p,
        # resolution = res[args.res], # float or array of x.shape
        rect_dim=-1,
        disp = bool(args.ver),
        locally_biased=False,
        restart='Results/History{}.csv'.format(scenario) if args.x == 1 else '',
        alt_threshold=1.25,
        program=(
            {'maxiter':np.inf,
              'resolution':res[0],
              'population':2*args.p,
              },
            {'maxiter':np.inf,
              'resolution':res[1],
              'population':int(1.5*args.p),
              },
            {'maxiter':np.inf,
              'resolution':res[2],
              'population':int(1.25*args.p),
              },
            {'maxiter':np.inf,
              'resolution':res[3],
              'population':args.p,
              },
            {'maxiter':20,
              'resolution':res[4],
              'population':args.p,
              },  
            {'maxiter':15,
              'resolution':res[5],
              'population':args.p,
              },
            )
        )

    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    print(result.x, result.f)
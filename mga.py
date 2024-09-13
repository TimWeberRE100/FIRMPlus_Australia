# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:17:48 2024

@author: u6942852
"""


import numpy as np
import datetime as dt
from numba import njit
from csv import writer

from Input import *
from spacepartition import Spacepartition



@njit
def gen_inds(npop, vsize=args.vp):
    range_gen = range(npop//vsize + 1) if npop%vsize != 0 else range(npop//vsize)
    indcs = [np.arange(n*vsize, min((n+1)*vsize, npop), dtype=np.int64) for n in range_gen]
    
    inds = []
    for i in range(len(indcs)):
        inds.append(indcs[i])
    return inds


@njit
def Obj(x):
    S = Solution(x)
    S._evaluate()
    result = np.array([S.LCOE + S.Penalties, 
                       S.LCOE, S.LCOG, S.LCOBS, 
                       S.LCOBT, S.LCOBL], dtype=np.float64)
    return result
    
    
@njit
def Vobj(x, maxvectorwidth=args.vp):
    results = np.empty((len(x.T), 6), dtype=np.float64)
    inds = gen_inds(len(x.T), maxvectorwidth)
    for ind in inds:
        S = VSolution(x[:, ind])
        S._evaluate()
        results[ind] = np.array([S.LCOE + S.Penalties, 
                           S.LCOE, S.LCOG, S.LCOBS, 
                           S.LCOBT, S.LCOBL])
    return results
    
if __name__ == '__main__':
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    
    z = (pzones+wzones+nodes)
    first_pass = np.array([10.1]*z + [500.1])
    ultralow_res = np.array([1.1]*z + [100]) # 1 GW, 100 GWh
    low_res = np.array([0.1]*z + [10.0]) # 100 MW, 10 GWh
    medium_res = np.array([0.01]*z + [1.0]) # 10 MW, 1 GWh
    high_res = np.array([0.001]*z + [0.01]) # 1 MW, 100 MWh
    ultrahigh_res = np.array([0.000_1]*z + [0.01]) # 0.1 MW, 10 MWh
    polishing = np.array([0.000_001]*z + [0.000_1]) # 1 kW, 100 kWh
    
    res = [first_pass, ultralow_res, low_res, medium_res, high_res, ultrahigh_res, polishing]

    problem = Spacepartition(
        func=Vobj if args.vec else Obj  , 
        bounds=(lb, ub)  ,
        f_args=(args.vp,) if args.vec else ()  ,
        printfile='Results/History{}'.format(scenario) if args.cb == 2 else ''  ,
        vectorizable=bool(args.vec)  ,
        max_dims= 6  ,
        disp = bool(args.ver)  ,
        restart='Results/History{}'.format(scenario) if args.resume == 1 else ''  ,
        near_optimal=1.05  ,
        nextras = 5,
        )

    problem.Step({'max_iter':25,
                  'max_res':res[0],
                  'near_optimal':np.inf, 
                  'max_pop':25,
                  })
    print('step2')
    problem.Step({'max_iter':10,
                  'max_res':res[0],
                  'near_optimal':1.5, 
                  'max_pop':25,
                  })
    print('step3')
    problem.Step({'max_iter':20,
                  'max_res':res[1],
                  'near_optimal':1.5, 
                  'max_pop':25,
                  })
    print('step4')
    problem.Step({'max_iter':np.inf,
                  'max_res':res[1],
                  'near_optimal':1.03, 
                  'max_pop':1000,
                  })
    print('step5')
    problem.Step({'max_iter':20,
                  'max_res':res[1],
                  'near_optimal':1.1, 
                  'max_pop':50,
                  })
    print('step6')
    problem.Step({'max_iter':np.inf,
                  'max_res':res[1],
                  'near_optimal':1.03, 
                  'max_pop':1000,
                  })
    print('polish')
    problem.Polish({'max_res':res[1], 
                    'near_optimal':1.03})
    
    result = problem.ReturnElite()


    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    print(result.x, result.f)
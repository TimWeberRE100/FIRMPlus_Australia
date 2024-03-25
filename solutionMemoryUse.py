# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:51:50 2024

@author: u6942852
"""

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', type=int,   required=True)
parser.add_argument('-p', type=int,   required=True)
parser.add_argument('-m', type=float, required=True)
parser.add_argument('-r', type=float, required=True)
parser.add_argument('-s', type=int,   required=True)
parser.add_argument('-cb',type=int,   required=True)
parser.add_argument('-v', type=int,   required=True)
parser.add_argument('-vp',type=int,   required=True)
parser.add_argument('-w', type=int,   required=True)
args = parser.parse_args()

from Input import *
from numpy import atleast_2d, array
from numpy.random import rand

width = args.vp

dvs = len(lb)

unnorm = atleast_2d(ub-lb)

def objective(x):
    S = Solution(x)
    return S.Lcoe + S.Penalties

if args.w == 1: 
    x = rand(width, dvs)*unnorm + atleast_2d(lb)
    x = x.T
    objective(x)

if args.w > 1:
    
    processes=args.w
    x = rand(width*processes, dvs)*unnorm + atleast_2d(lb)
    x = x.T

    npop = x.shape[1]
    
    vsize = npop//processes + 1 if npop%processes != 0 else npop//processes
    vsize = min(vsize, args.vp, npop)
    r = range(npop//vsize + 1) if npop%vsize != 0 else range(npop//vsize)
    
    arrs = [x[:, n*vsize: min((n+1)*vsize, npop)] for n in r]
    
    from multiprocessing import Pool
    
    with Pool(processes=args.w) as processPool: 
        result = processPool.map(objective, arrs)
    result = array(result)
        
        
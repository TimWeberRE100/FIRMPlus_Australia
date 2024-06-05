# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import datetime as dt
from numba import jit, float64, cuda, guvectorize
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', default=1000, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=100, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')
parser.add_argument('-s', default=11, type=int, required=False, help='11, 12, 13, ...')
parser.add_argument('-n', default='Super1', type=str, required=False, help='node=Super1')
args = parser.parse_args()

scenario = args.s
node = args.n

from Input import *


@jit(float64(float64[:]))
def Obj(x):
    S = Solution(x)
    S._evaluate()
    result = S.Lcoe + S.Penalties
    return result

@guvectorize([(float64[:,:], float64[:])], '(m, n)->(m)')
def parallel_objs(x, result):
    for i in range(x.shape[0]):
        result[i] = Obj(x[i])
    
#%%
nvec = 5
input_vector = (np.random.rand(nvec, len(lb))*(ub-lb)+lb)
# result = np.empty(nvec, dtype=np.float64)
input_vector.shape
result = parallel_objs(input_vector)

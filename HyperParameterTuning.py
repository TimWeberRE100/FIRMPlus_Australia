# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:51:27 2024

@author: u6942852
"""

import numpy as np
import psutil as ps
from sys import platform
from subprocess import Popen, CalledProcessError, STDOUT
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta

from modified_memory_profiler import memory_usage
from Optimisation import args
# from Input import * 

VERBOSE=True

# Find maximum number of solution which can be run at a time while keeping free 
# a reserve of 10 %
# dvs = len(lb)

# unnorm = np.atleast_2d(np.array(ub)-lb)
# lb = np.atleast_2d(lb)

ram = ps.virtual_memory()[0] #+ ps.swap_memory()[0]
# set max at 85% - bear in mind that there is memory use from other python and
# system parameters. There is also swap memory, but we want to limit use of it for speed
memory_budget = 0.85
budget = memory_budget*ram/1024//1024

print(f"memory budget = {budget}/{ram/1024//1024} MiB")

## Yet to be tested
if platform=='linux':
    import resource
    limit = int(memory_budget*ram)
    resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))
    print(f"memory resource limit set to {budget}/{ram/1024//1024} MiB")
    

def _memory_test_run(width, rec=0, rec_lim=3, include_children=False):
    # There is a less annoying way to do this, but it was freezing on my computer 
    def _recurse():
        if rec < rec_lim:
            if VERBOSE: print(f'  subprocess error: trying step again (attempt {rec+1}/{rec_lim})')
            return _memory_test_run(width, rec+1, rec_lim)
        else: 
            return None, None
    args.vp = width
    r = [x for xs in [('-'+n, str(v)) for n,v in args._get_kwargs()] for x in xs]

    try:  
        mem, retcode = memory_usage(proc=Popen(['python3','solutionMemoryUse.py']+r),
                           interval=0.01, max_usage=True,retval=True, include_children=include_children)
    
    except ps.NoSuchProcess: # implementation is a bit unstable? this helps
        return _recurse()
    if retcode != 0: 
        return _recurse()
    
    return mem, (rec+1)

def iterate_memory_test(start, end, inc, prev_mem=None, include_children=False):
    assert (end-start)%inc == 0

    timesum = timedelta(0)
    for i in range(start, end, inc):
        tstart = datetime.now()
        mem, rec = _memory_test_run(i, 0 , 2, include_children)
        if mem is None: 
            break
        timesum += (datetime.now()-tstart)/rec
        if VERBOSE: 
            print(f"width: {i}, mem: {round(100*mem/budget, 2)} % of budget. (took: {(datetime.now()-tstart)/rec})")
        if prev_mem is not None: 
            est_next = 2*mem-prev_mem
            if est_next > budget*1.1:
                if VERBOSE: 
                    print(f"width: {i}, next increment {i+inc} expected to exceed budget at {round(100*est_next/budget, 2)} %")
                    i+=inc
                break
        if mem > budget: 
            break
        prev_mem=mem
        
    return i-inc, timesum*inc/i, prev_mem

#%%
W, args.w = args.w, 1
# if VERBOSE: 
#     print("Beginning test for vectorisation width on a single core.\n","-"*70)
# n=0
# prev_mem = None
# for p in range(8, 4, -1):
#     n, avetime, prev_mem = iterate_memory_test(n+pow(10, p-1), n+pow(10, p), pow(10, p-1), prev_mem)

    
# print("Maximum no. solutions to be run together on a single core: ",n)
# print("Approximate time to run this order of magnitude of solutions together", avetime)

#%%
args.w = 2 

if VERBOSE: 
    print("Beginning test for vectorisation width on two cores.\n","-"*70)
n=0
prev_mem = None
for p in range(8, 3, -1):
    n, avetime, prev_mem = iterate_memory_test(n+pow(10, p-1), n+pow(10, p), pow(10, p-1), prev_mem, include_children=True)
    
print("Maximum no. solutions to be run together on two cores: ", n)
print("Approximate time to run this order of magnitude of solutions together", avetime)


#%%
# test_solution(x.T)

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

ram = ps.virtual_memory()[0] #+ ps.swap_memory()[0]
# set max at 85% - bear in mind that there is memory use from other python and
# system parameters. There is also swap memory, but we want to limit use of it for speed
memory_budget = 0.85
budget = memory_budget*ram/1024//1024

print(f"memory budget = {budget}/{ram/1024//1024} MiB")

pycmd = 'python'
if platform=='linux':
    pycmd = 'python3'
    import resource
    limit = int(memory_budget*ram)
    resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))
    print(f"memory resource limit set to {budget}/{ram/1024//1024} MiB")
    
#%% Memory Testing
def _memory_test_eval(width, rec=0, rec_lim=3, include_children=False):
    # There is a less annoying way to do this, but it was freezing on my computer 
    def _recurse():
        if rec < rec_lim:
            if VERBOSE: print(f'  subprocess error: trying step again (attempt {rec+1}/{rec_lim})')
            return _memory_test_eval(width, rec+1, rec_lim)
        else: 
            return None, None
    args.vp = width
    r = [x for xs in [('-'+n, str(v)) for n,v in args._get_kwargs()] for x in xs]

    try:  
        mem, retcode = memory_usage(proc=Popen([pycmd,'solutionMemoryUse.py']+r),
                           interval=0.0001, max_usage=True,retval=True, include_children=include_children)
    
    except ps.NoSuchProcess: # implementation is a bit unstable? this helps
        return _recurse()
    if retcode != 0: 
        return _recurse()
    
    return mem, (rec+1)

def _memory_test_it(start, end, inc, prev_mem=None, include_children=False):
    assert (end-start)%inc == 0

    timesum = timedelta(0)
    for i in range(start, end, inc):
        tstart = datetime.now()
        mem, rec = _memory_test_eval(i, 0 , 2, include_children)
        if mem is None: 
            break
        timesum += (datetime.now()-tstart)/rec
        if VERBOSE: 
            print(f"width: {i}, mem: {round(100*mem/budget, 2)} % of budget. (took: {(datetime.now()-tstart)/rec})")
        if mem > budget: 
            break
        if prev_mem is not None: 
            est_next = 2*mem-prev_mem
            if est_next > budget*1.1:
                if VERBOSE: 
                    print(f"width: {i}, next increment {i+inc} expected to exceed budget at {round(100*est_next/budget, 2)} %")
                    i+=inc
                break
        prev_mem=mem
        
    return i-inc, timesum*inc/i, prev_mem

def memory_test(workers = 1):
    W, args.w = args.w, workers
    if VERBOSE: 
        print("Beginning test for vectorisation width on {workers} core(s).\n{'-'*70}")
    n=0
    prev_mem = None
    for p in range(5, -1, -1):
        n, avetime, prev_mem = _memory_test_it(n+pow(10, p-1), n+pow(10, p), pow(10, p-1), prev_mem)
    
    print(f"Maximum no. solutions to be run together on {workers} core(s): {n}")
    args.w = W






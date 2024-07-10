# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:30:19 2024

@author: u6942852
"""

from Input import scenario, lb, ub
import warnings
import numpy as np
import pandas as pd
from numba import njit, prange#, float64, int64
# from numba.experimental import jitclass

from DirectAlgorithm import hyperrectangle, _factor2, hrects_border, _child_loop, _reconstruct_from_centre

import seaborn as sns
import matplotlib.pyplot as plt

def readin(file, bounds):
    history = np.genfromtxt(file+'-children.csv', delimiter=',', dtype=np.float64)
    try: 
        minima = np.genfromtxt(file+'-resolved.csv', delimiter=',', dtype=np.float64)
        if len(minima) != 0:
            history = np.vstack((history, np.atleast_2d(minima)))
    except FileNotFoundError:
        pass
    try: 
        parents = np.genfromtxt(file+'-parents.csv', delimiter=',', dtype=np.float64)
        pmin, pminidx = parents[:,0].min(), parents[:,0].argmin()
    except FileNotFoundError:
        pmin, pminidx= np.inf, None
        warnings.warn("Warning: No parents file found.", UserWarning)
    
    fs, xs = history[:,:3], history[:,3:]
    
    xs, lbs, ubs = _reconstruct_from_centre(xs, bounds)
    
    if fs[:,0].min() < pmin:
        elite = fs[:,0].argmin()
        elite = hyperrectangle(xs[elite], *fs[elite], lbs[elite], ubs[elite], np.nan)
    else: 
        fps, xps = parents[:,:3], parents[:,3:]
        xps, lbps, ubps = _reconstruct_from_centre(np.atleast_2d(xps[pminidx, :]), bounds)
        elite = hyperrectangle(xps[0,:], *fps[pminidx,:], lbps[0,:], ubs[0,:], np.nan)
        del fps, xps, lbps, ubps, parents
        
# =============================================================================
#     # This assumes direct didn't fully finish
#     _child_loop(xs[:2,:], fs[:2,:], lbs[:2, :], ubs[:2, :]) # compile jit
#     archive = _child_loop(xs, fs, lbs, ubs)
# =============================================================================
    
    # Assumes all near-optimal rectangles are at maximum resolution
    archive = [hyperrectangle(xs[i], *fs[i], lbs[i], ubs[i], np.nan) for i in range(len(fs))]
    return archive, elite



@njit(parallel=True)
def islanded(archive, pool, antipool, highest_res=False):
    accepted = np.empty(len(pool), dtype=np.bool_)
    if len(pool) < len(antipool) or not highest_res:
        for i in prange(len(pool)):
            accepted[i] = _islandborder(archive[pool[i]], archive, pool)
    else: # if len(pool) >= len(antipool) and highest_res:
        for i in prange(len(pool)):
            accepted[i] = _borderbysum(archive[pool[i]], archive, antipool)
    return pool[accepted]

@njit 
def _borderbysum(h, archive, secondpool):
    """ Returns True if h is landlocked by secondpool """
    """ Assumes h is of the smallest resolution in archive """
    faces = h.ndim*2 - (h.ub == ub).sum() - (h.lb == lb).sum()
    for j in secondpool:
        h2 = archive[j]
        if hrects_border(h, h2):
            faces -= 1 
        if faces == 0:
            return True
    return False

@njit
def _islandborder(h, archive, secondpool):
    """ Returns True if h is islanded from secondpool """
    for j in secondpool:
        h2 = archive[j]
        if hrects_border(h, h2):
            return False
    return True

@njit(parallel=True)
def landlocked(archive, pool, antipool, highest_res=False):
    accepted = np.empty(len(pool), dtype=np.bool_)
    if len(pool) >= len(antipool) or not highest_res:
        for i in prange(len(pool)):
            accepted[i] = _islandborder(archive[pool[i]], archive, antipool)
    else: # if len(pool) > len(antipool) and highest_res:
        for i in prange(len(pool)):
            accepted[i] = _borderbysum(archive[pool[i]], archive, pool)
    return pool[accepted]
        

def nearoptimal_volume(no_ll, no_nll, nno_nll):
    #volume of landlocked near optimal points
    vol_no_ll = sum([archive[i].volume for i in no_ll]) 
    #volume of near-optimal points which border a non-near-optimal point
    vol_no_nll = sum([archive[i].volume for i in no_nll])
    #volume of non-near-optimal points which border a near-optimal point
    vol_nno_nll = sum([archive[i].volume for i in nno_nll])
    
    # assuming planar true boundary, up to 50% of a near-optimal rectangle may be 
    # non-near-optimal space
    vol_lb = vol_no_ll + 0.5*vol_no_nll
    # best guess at volume is naive of boundary effects
    vol_c = vol_no_ll + vol_no_nll
    # assuming planar true boundary, up to 50% of a non-near-optimal rectangle may  
    # be near-optimal space
    vol_ub = vol_no_ll + vol_no_nll + 0.5*vol_nno_nll
                
    return vol_lb, vol_c, vol_ub


#%%
if __name__=='__main__': 
    
    ##TODO function to query for non-dominated points of specified variables
    
    file = 'Results/History{}'.format(scenario)    
    archive, elite = readin(file, (lb, ub))
    raise KeyboardInterrupt
    assert abs(sum([h.volume for h in archive]) - (ub-lb).prod()) < 0.1
    #assume direct exploration went to full resolution
    cuts = [h.cuts for h in archive]
    cuts = max(cuts)
    archive = [h for h in archive if h.cuts <= cuts]
    slack = 1.05
    threshold = slack*elite.f
    
    # mask of near-optimal points
    no = np.array([i for i, h in enumerate(archive) if h.f<threshold])
    # indices of non-near-optimal points
    nno = np.setdiff1d(np.arange(len(archive)), no, assume_unique=True)
    # indices of near-optimal points which border only near-optimal points
    no_ll = landlocked(archive, no, nno, True)
    # indices of near-optimal points which border at least one non-near-optimal point
    no_nll = np.setdiff1d(no, no_ll)
    # indices of non-near-optimal points which border only non-near-optimal points
    nno_ll = landlocked(archive, nno, no, True)
    # indices of non-near-optimal points which border at least one near-optimal point
    nno_nll = np.setdiff1d(nno, nno_ll)
    
    vol_lb, vol_c, vol_ub = nearoptimal_volume(no_ll, no_nll, nno_nll)
    
    raise KeyboardInterrupt
    
    # indices of near-optimal points which do not border another
    no_isl = islanded(archive, no, np.empty([], np.float64), True)
    # indices of non-near-optimal points which do not border another
    nno_isl = islanded(archive, nno)
    
    centres = np.array([h.centre for h in archive if h.f < threshold])
    
    centres = (centres-lb)/(ub-lb)
    
    #assuming even sampling
    centroid = centres.mean(axis=0)
    
    distances = ((centres - centroid)**2).sum(axis=1)**(1/2)
    
    inc = 10000
    incs = np.linspace(0, distances.max(), inc+1)
    vols = np.empty(inc)
    for i in range(inc):
        vols[i] = 4 / 3 * np.pi * (incs[i+1]**3 - incs[i]**3)
    
    counts = np.zeros(inc, np.float64)
    distances = np.sort(distances)
    j=0
    for i in range(inc): 
        while distances[j] < incs[i+1]:
            counts[i] += 1 
            j+=1
    
    fig, ax = plt.subplots(dpi=200)
    sns.kdeplot( 
        x=counts/vols,
        # x=incs[1:],
        ax=ax,
        # estimator='mean'
        )
    
    distvol = distances / (4* np.pi * distances ** 2) 
    
    fig, ax = plt.subplots(dpi=200)
    sns.kdeplot( 
        x=distvol,
        # x=incs[1:],
        ax=ax,
        # estimator='mean'
        )
    
    
    # i=0
    # weights=[]
    # for j in range(inc-1): 
    #     while i < len(distances) and distances[i] <= incs[j+1]:
    #         weights.append(vols[j])
    #         i+=1
    # weights = np.array(weights)
    
    fig, ax = plt.subplots()
    plot = sns.histplot(
        x=distances, 
        bins=incs[1:],
        # weights=1/distances,
        ax=ax,
        )
    
    c = pd.DataFrame(centres, columns=list(range(centres.shape[1])))
    c['cost'] = pd.Series([h.f for i, h in enumerate(archive) if i in no])
    sns.pairplot(
        c,
        kind='hist',
        )


    pd.Series(distances).describe()
    
    
    ##TODO 
    # solution density = no. of point theoretically in that shell / no. in the shell
    
    

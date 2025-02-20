# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:48:18 2024

@author: u6942852
"""

import datetime as dt
import csv
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt


from Input import *
from optimisation_utils import Jacobian


@njit 
def objective(x):
    S = Solution(x)
    S._evaluate()
    return S.Lcoe, S.Penalties

@njit 
def objective_lcoe(x):
    S = Solution(x)
    S._evaluate()
    return S.Lcoe

@njit(parallel=True)
def sample_objective(x, n, vals):
    x[n] = 0 
    nmask = np.zeros(len(x))
    nmask[n] = 1 
    lcoes, penalties = np.empty(len(vals), np.float64), np.empty(len(vals), np.float64)
    for i in prange(len(vals)):
        lcoes[i], penalties[i] = objective(x+nmask*vals[i])
    return lcoes, penalties


def normalise(x):
    return (x-lb)/(ub-lb)

def unnormalise(x):
    return x*(ub-lb) + lb

def relative_difference(a,b):
    return (2*(a-b)/(a+b))

def check_nonconvex():
    x0 = unnormalise(np.random.rand(len(lb)))
    convex = True
    counter = 0
    points=[x0]
    while convex:
        x1 = unnormalise(np.random.rand(len(lb)))
        points.append(x1)
        # https://math.stackexchange.com/questions/3325382/how-to-check-if-a-function-is-convex
        f_ave = objective_lcoe((x0 + x1)/2)
        ave_f = (objective_lcoe(x0) + objective_lcoe(x1))/2
        
        if convex:=f_ave - ave_f <= 0:
            x0 = x1.copy()
        else: 
            print(counter, '\nnon-convex', 'f(ave):', f_ave, 'ave(f,f):', ave_f)
        if counter% 10 == 0:
            print(counter, ', ', sep='', end = '')
        counter+=1 
        
def plot_variable_sweep(n, x, samples, ax=None, bounds='auto'):
    ax = plt.gca() if ax is None else ax
    fig = plt.gcf()
    ax2 = plt.twinx()
    
    l, u = (lb[n], ub[n]) if bounds == 'auto' else bounds
    
    vals = np.linspace(l, u, num=samples)
    
    lcoes, penalties = sample_objective(x, n, vals)
    
    ax.plot(vals, lcoes, color='blue', label='LCOE')
    ax2.plot(vals, penalties, color='orange', label='Penalties')
    fig.legend()
    ax.set_ylabel('LCOE ($/MWh)')
    ax2.set_ylabel('Penalties')
    source = 'pv' if n < pidx else 'wind' if n < widx else 'php' if n < spidx else 'phs' if n < seidx else 'hvdc'
    unit = 'GWh' if source == 'phs' else 'GW' 
    ax.set_xlabel(f'Variable {n} ({source} {unit})')
    ax.set_title(f'Sweep of variable {n} ({source}). Others held constant near optimal.')
    
if __name__ == '__main__':
    
    # check_nonconvex()
    np.set_printoptions(suppress=True)
    x = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
    
    fig, ax = plt.subplots()
    plot_variable_sweep(0, x.copy(), 50)
    fig, ax = plt.subplots()
    plot_variable_sweep(pidx, x.copy(), 50)
    # fig, ax = plt.subplots()
    # plot_variable_sweep(widx, x.copy(), 50)
    # fig, ax = plt.subplots()    
    # plot_variable_sweep(26, x.copy(), 50)
    fig, ax = plt.subplots()
    plot_variable_sweep(spidx, x.copy(), 50)
    # fig, ax = plt.subplots()
    # plot_variable_sweep(seidx, x.copy(), 50)

    # fig, ax = plt.subplots()
    # plot_variable_sweep(0, x.copy(), 200, ax=ax, bounds=(0,7.5))
    # fig, ax = plt.subplots()
    # plot_variable_sweep(1, x.copy(), 100, ax=ax, bounds=(0,1.))
    # fig, ax = plt.subplots()
    # plot_variable_sweep(pidx, x.copy(), 100, ax=ax, bounds=(0,0.15))
    # fig, ax = plt.subplots()
    # plot_variable_sweep(17, x.copy(), 100, ax=ax, bounds=(0,2.))


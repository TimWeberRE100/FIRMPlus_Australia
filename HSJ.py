# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:23:48 2024

@author: u6942852
"""


from datetime import datetime as dt
import csv
import numpy as np
import pyomo.environ as pyo

np.set_printoptions(suppress=True)

from Input import * 
from Weightings import *

from Optimisation import instantiate_model, cost_optimise

#%%
def initiate_printfile():
    def print_header():
        with open(f'Results/HSJ{scenario}.csv', 'w', newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Iteration","LCOE","LCOG","LCOBS","LCOBT","LCOBL","PV (TWh p.a.)","Wind (TWh p.a.)","Hydro&Bio (TWh p.a.)"]+
                                [f"CPV{n}" for n in range(npv)]+
                                [f"CWind{n}" for n in range(nwind)] + [f"CPHP{n}" for n in range(nodes)]+
                                [f"CPHE{n}" for n in range(nodes)] + [f"CHvdc{n}" for n in range(nhvdc)])
    try:
        print_header()
    except FileNotFoundError:
        from os import mkdir
        mkdir('Results')
        del mkdir
        print_header()

def printout(model, it):
# =============================================================================
#     Would be nice to add more stats 
# Maybe jit some stat calculations? 
# =============================================================================
    S = Solution(model)
    with open(f'Results/HSJ{scenario}.csv', 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([it, S.LCOE, S.LCOG, S.LCOBS, S.LCOBT, S.LCOBL, S.GPV, S.GWind, S.GHydro+S.GBio]+
                            list(S.cpv)+list(S.cwind)+list(S.cphp)+list(S.cphe)+list(S.chvdc))

class generate_alternatives:
    def __init__(self, model, strategy=random_weighting, strategy_args=()):
        self.model = model
        self.strategy = strategy
        self.strategy_args = strategy_args
        self.optimum = pyo.value(self.model.LCOE)

        initiate_printfile()
        printout(model, 'optimal')
    
    def generate(self, threshold, nhops):
        self.optimiser = pyo.SolverFactory('gurobi_persistent')
        
        self.near_optimal = self.optimum*threshold
        self.model.near_optimal = pyo.Constraint(rule=lambda m:m.LCOE <= self.near_optimal)
        
        self.model.WPV   = pyo.Var(self.model.pvl,   domain=pyo.Reals, initialize = dict(zip(range(1, npv+1),   (0,)*npv)))
        self.model.WWind = pyo.Var(self.model.windl, domain=pyo.Reals, initialize = dict(zip(range(1, nwind+1), (0,)*nwind)))
        self.model.WPHP  = pyo.Var(self.model.nodes, domain=pyo.Reals, initialize = dict(zip(range(1, nodes+1), (0,)*nodes)))
        self.model.WPHE  = pyo.Var(self.model.nodes, domain=pyo.Reals, initialize = dict(zip(range(1, nodes+1), (0,)*nodes)))
        self.model.WHvdc = pyo.Var(self.model.lines, domain=pyo.Reals, initialize = dict(zip(range(1, nhvdc+1), (0,)*nhvdc)))
        
        self.model.wsum = pyo.Objective(rule=lambda m: pyo.summation(m.WPV,   m.cpv)  + 
                                                       pyo.summation(m.WWind, m.cwind)+
                                                       pyo.summation(m.WPHP,  m.cphp) + 
                                                       pyo.summation(m.WPHE,  m.cphe) + 
                                                       pyo.summation(m.WHvdc, m.chvdc),
                                        sense=pyo.maximize)
        
        self.optimiser.set_instance(self.model)
        
        # variables to be fixed before calling set_instance
        ## See: https://groups.google.com/g/pyomo-forum/c/GceDX47MpPQ
        for v in self.model.WPV.values():
            v.fix()
            self.optimiser.update_var(v)
        for v in self.model.WWind.values():
            v.fix()
            self.optimiser.update_var(v)
        for v in self.model.WPHP.values():
            v.fix()
            self.optimiser.update_var(v)
        for v in self.model.WPHE.values():
            v.fix()
            self.optimiser.update_var(v)
        for v in self.model.WHvdc.values():
            v.fix()
            self.optimiser.update_var(v)
        
        for hop in range(nhops):
            start=dt.now()
            print(f"HSJ {hop} starts: {start}", end='')

            weights = self.strategy(self.model, *self.strategy_args)
            self._iterate(weights, hop)
            printout(model, hop)
            
            end=dt.now()
            print(f". Took: {end-start}")

    def _iterate(self, weights, hop):
        
        for i, v in enumerate(self.model.WPV.values()):
            v.value = weights[i]
            self.optimiser.update_var(v)
        for i, v in enumerate(self.model.WWind.values()):
            v.value = weights[i+pidx]
            self.optimiser.update_var(v)
        for i, v in enumerate(self.model.WPHP.values()):
            v.value = weights[i+widx]
            self.optimiser.update_var(v)
        for i, v in enumerate(self.model.WPHE.values()):
            v.value = weights[i+spidx]
            self.optimiser.update_var(v)
        for i, v in enumerate(self.model.WHvdc.values()):
            v.value = weights[i+seidx]
            self.optimiser.update_var(v)

        self.optimiser.solve(self.model, save_results=True)


if __name__ == '__main__':
    model = instantiate_model()
    model = cost_optimise(model)
    
    # strategy = evolving_average()
    strategy = random_weighting
    hsj = generate_alternatives(model, strategy)
    hsj.generate(1.05, 50)


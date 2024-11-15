# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:41:12 2024

@author: u6942852
"""

import numpy as np
np.set_printoptions(suppress=True)

from Input import * 

class evolving_average:
    # https://www-sciencedirect-com.virtual.anu.edu.au/science/article/pii/S0306261923003665
    def __init__(self):
        self.weights = np.zeros(npv+nwind+nodes+nodes+nhvdc, float)
        self.x_avg = np.ones_like(self.weights)
        
        self.n=0
        
    def __call__(self, model):
        self.n+=1
        
        x_inst = np.concatenate((
            np.array([cpv.value   for cpv   in model.cpv.values()  ]).round(5),
            np.array([cwind.value for cwind in model.cwind.values()]).round(5),
            np.array([cphp.value  for cphp  in model.cphp.values() ]).round(5),
            np.array([cphe.value  for cphe  in model.cphe.values() ]).round(5),
            np.array([chvdc.value for chvdc in model.chvdc.values()]).round(5),
            ))
        
        self.weights = np.abs((self.x_avg * self.n - x_inst) / self.x_avg)

        self.x_avg = (self.x_avg * self.n + x_inst) / (self.n+1)
        
        return self.weights
    
def random_weighting(model):
    return np.random.rand(xlen)
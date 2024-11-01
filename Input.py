# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
import pyomo.environ as pyo
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-s', default=21, type=int, required=False, help='scenario')
parser.add_argument('-y', default=1, type=int, required=False, help='no. of years')
args = parser.parse_args()

scenario = args.s
years = args.y

Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
PVl =   np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)

n_node = dict((name, i) for i, name in enumerate(Nodel))
Nodel_int, PVl_int, Windl_int = (np.array([n_node[node] for node in x], dtype=np.int64) for x in (Nodel, PVl, Windl))
Nodel_int, PVl_int, Windl_int = (x.astype(np.int64) for x in (Nodel_int, PVl_int, Windl_int))

resolution = 0.5 # timestep resolution in hours
StartCharge = 0.5 # starting level of storage energy

MLoad = np.genfromtxt('Data/electricity.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel))) 

TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) 
TSWind = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Windl))) 

assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(np.float64)
CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] 

# FQ, NQ, NS, NV, AS, SW, only TV constrained
DCloss = np.array([1500, 1000, 1000, 800, 1200, 2400, 400, 
                   700, 
                   ]) * 0.03 * pow(10, -3)

efficiency = 0.8 # round trip efficiency of storage
factor = np.genfromtxt('Data/factor.csv', delimiter=',', usecols=1)

network = np.array([[0, 3], #FNQ-QLD
                    [1, 3], #NSW-QLD
                    [1, 4], #NSW-SA
                    [1, 6], #NSW-VIC
                    [2, 4], #NT-SA
                    [4, 7], #SA-WA
                    [5, 6], #TAS-VIC
                    [4, 6], #SA-VIC
                    ], dtype=np.int64)
    
if scenario<=17:
    node = Nodel[scenario % 10]

    MLoad = MLoad[:, Nodel==node]
    TSPV = TSPV[:, PVl==node]
    TSWind = TSWind[:, Windl==node]
    CHydro, CBio = [x[Nodel==node] for x in (CHydro, CBio)]

    Nodel_int, PVl_int, Windl_int = [x[x==n_node[node]] for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [x[x==node] for x in (Nodel, PVl, Windl)]
    network_mask = np.zeros(7, bool)
    network = np.empty((0,0), int)

elif scenario>=21:
    coverage = [np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])][scenario % 10 - 1] 
    
    MLoad = MLoad[:, np.in1d(Nodel, coverage)]
    TSPV = TSPV[:, np.in1d(PVl, coverage)]
    TSWind = TSWind[:, np.in1d(Windl, coverage)]
    CHydro, CBio = [x[np.in1d(Nodel, coverage)] for x in (CHydro, CBio)]
    
    if 'FNQ' not in coverage:
        MLoad[:, np.where(coverage=='QLD')[0][0]] /= 0.9
        
    coverage_int = np.array([n_node[node] for node in coverage])
    Nodel_int, PVl_int, Windl_int = [x[np.isin(x, coverage_int)] for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [x[np.isin(x, coverage)] for x in (Nodel, PVl, Windl)]

    network_mask = np.array([(network==j).sum(axis=1).astype(np.bool_) for j in Nodel_int]).sum(axis=0)==2
    network = network[network_mask,:]
    networkdict = {v:k for k, v in enumerate(Nodel_int)}
    #translate into indicies rather than Nodel_int values
    network = np.array([networkdict[n] for n in network.flatten()], np.int64).reshape(network.shape)


if scenario >= 31:
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    
    TSPV = np.stack([TSPV[:, PVl==node].mean(axis=1) for node in coverage]).T
    TSWind = np.stack([TSWind[:, Windl==node].mean(axis=1) for node in coverage]).T
    # having full of zeros and setting lb,ub=0,0 makes code faster
    TSPV = np.nan_to_num(TSPV, False, 0)
    warnings.simplefilter('default', RuntimeWarning)
    
    Nodel_int, PVl_int, Windl_int = [np.unique(x) for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [np.unique(x)  for x in (Nodel, PVl, Windl)]
    
intervals, nodes = MLoad.shape
pzones, wzones = (TSPV.shape[1], TSWind.shape[1])
pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes)

energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

firstyear = 2020
finalyear = firstyear+years-1

nhvdc = network_mask.sum()

MLoad = MLoad / 1000. #MW to GW

def countleaps(startyear, finalyear):
    leaps=0
    for y in range(startyear, finalyear+1):
        if y%4 == 0 and y%1000 != 0: 
            leaps+=1
    return leaps
    

#%%
class Solution:
    def __init__(self, model, years=years, leapdays=True):
        self.scenario, self.nodes = scenario, nodes
        self.Nodel, self.PVl, self.Windl = Nodel, PVl, Windl
        
        self.network, self.network_mask, self.DCloss = network, network_mask, DCloss[network_mask]
        self.pos_export_lines = [np.where(network[:,0]==n)[0] for n in range(nodes)] # pyomo uses 1-indexing
        self.neg_export_lines = [np.where(network[:,1]==n)[0] for n in range(nodes)] # pyomo uses 1-indexing

        self.firstyear, self.years = firstyear, years
        self.finalyear = self.firstyear+self.years-1
        self.resolution = resolution
        leaps = countleaps(self.firstyear, self.finalyear) if leapdays else 0
        self.intervals = int((years*365 + leaps)*24/resolution)
        
        self.StartCharge, self.efficiency = StartCharge, efficiency
        
        # capacities in GW and GWh
        self.cpv =   np.array([model.cpv[i].value   for i in model.cpv])
        self.cwind = np.array([model.cwind[i].value for i in model.cwind])
        self.cphp =  np.array([model.cphp[i].value  for i in model.cphp])
        self.cphe =  np.array([model.cphe[i].value  for i in model.cphe]) 
        self.chvdc = np.array([model.chvdc[i].value for i in model.chvdc])
        self.chydro = CHydro
        self.cbio = CBio

        # operations in MW and MWh
        self.Discharge = np.array([model.discharge[i].value for i in model.discharge]).reshape(-1, nodes) * 1000.  #GW to MW
        self.Charge =    np.array([model.charge[i].value    for i in model.charge   ]).reshape(-1, nodes) * 1000.
        self.Storage =   np.array([model.storage[i].value   for i in model.storage  ]).reshape(-1, nodes) * 1000.  
        self.Hydro =     np.array([model.hydro[i].value     for i in model.hydro    ]).reshape(-1, nodes) * 1000. 
        self.Bio =       np.array([model.bio[i].value       for i in model.bio      ]).reshape(-1, nodes) * 1000.
        Hvdc_pos =       np.array([model.hvdc_pos[i].value  for i in model.hvdc_pos ]).reshape(-1, nhvdc) * 1000. 
        Hvdc_neg =       np.array([model.hvdc_neg[i].value  for i in model.hvdc_neg ]).reshape(-1, nhvdc) * 1000. 
        self.Hvdc = Hvdc_pos - Hvdc_neg

        self.Transmission = np.empty_like(self.Charge)
        for t in range(self.Charge.shape[0]):
            for n in range(self.Charge.shape[1]):
                self.Transmission[t, n]= (
                    + sum((Hvdc_pos[t, l] - Hvdc_neg[t, l]*(1-self.DCloss[l]) for l in self.pos_export_lines[n]))
                    + sum((Hvdc_neg[t, l] - Hvdc_pos[t, l]*(1-self.DCloss[l]) for l in self.neg_export_lines[n]))
                )
                
        self.Transmission * 1000.  
        self.PV = self.cpv*TSPV[:self.intervals, :] * 1000.
        self.Wind = self.cwind*TSWind[:self.intervals, :] * 1000.
        self.PV   = np.stack([self.PV[:,   np.where(self.PVl  ==node)[0]].sum(axis=1) for node in self.Nodel]).T
        self.Wind = np.stack([self.Wind[:, np.where(self.Windl==node)[0]].sum(axis=1) for node in self.Nodel]).T
        self.Load = 1000. * MLoad[:self.intervals, :]
        
        self.Spillage = -np.minimum(0, self.Load + self.Charge - self.Discharge - self.Hydro
                         - self.Bio - self.PV - self.Wind + self.Transmission)
        
        self.OBJ = pyo.value(model.OBJ)
# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
import pyomo.environ as pyo
from argparse import ArgumentParser

from Costs import cost_factors

np.set_printoptions(suppress=True)

parser = ArgumentParser()
parser.add_argument('-s', default=21,      type=int, required=False, help='scenario')
parser.add_argument('-c', default='csiro', type=str, required=False, help='cost source for pv/wind = csiro|irena')
parser.add_argument('-y', default=1,       type=int, required=False, help='no. of years')
args = parser.parse_args()

scenario = args.s
years = args.y
cost_source = args.c.lower()

Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
PVl   = np.array(['NSW']*9 + ['FNQ']*5 + ['QLD']*4 + ['SA']*9 + ['TAS']*3 + ['VIC']*6)
OnsWl = np.array(['NSW']*9 + ['FNQ']*5 + ['QLD']*4 + ['SA']*9 + ['TAS']*3 + ['VIC']*6)
OffWl = np.array(['NSW']*2 + ['SA']*1 + ['TAS']*2 + ['VIC']*2)

n_node = dict((name, i) for i, name in enumerate(Nodel))
nodesupportl = np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC'])
nodesupport = len(np.setdiff1d(Nodel, nodesupportl))

Nodel_int, PVl_int, OnsWl_int, OffWl_int = (np.array([n_node[node] for node in x], dtype=np.int64) for x in (Nodel, PVl, OnsWl, OffWl))
Nodel_int, PVl_int, OnsWl_int, OffWl_int = (x.astype(np.int64) for x in (Nodel_int, PVl_int, OnsWl_int, OffWl_int))

resolution = 0.5 # timestep resolution in hours
StartCharge = 0.5 # starting level of storage energy

MLoad = np.genfromtxt('Data/electricity.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel)-nodesupport)) # EOLoad(t, j), MW
#behind the meter solar 
MPVnsg = np.genfromtxt('Data/non-scheduled_pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel)-nodesupport))

TSPV   = np.genfromtxt('Data/utility_pv.csv',     delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) # TSPV(t, i), MW
TSOnsW = np.genfromtxt('Data/onshore_high.csv',   delimiter=',', skip_header=1, usecols=range(4, 4+len(OnsWl))) # TSWind(t, i), MW
TSOffW = np.genfromtxt('Data/offshore_fixed.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(OffWl))) # TSWind(t, i), MW

assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(np.float64)
CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] 

# FQ, NQ, NS, NV, AS, SW, only TV constrained
DClengths = np.array([1500, 1000, 1000, 800, 1200, 2400, 400, 700]) 
DCloss = DClengths * 0.03 * pow(10, -3)
undersea_mask = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)

network = np.array([['FNQ', 'QLD'], #FNQ-QLD
                    ['NSW', 'QLD'], #NSW-QLD
                    ['NSW', 'SA' ], #NSW-SA
                    ['NSW', 'VIC'], #NSW-VIC
                    ['NT',  'SA' ], #NT-SA
                    ['SA',  'WA' ], #SA-WA
                    ['TAS', 'VIC'], #TAS-VIC
                    ['SA',  'VIC'], #SA-VIC
                    ])
networksupport_mask = np.array([node in nodesupportl for node in network.ravel()]).reshape(network.shape).prod(axis=1).astype(np.bool_)
DClengths, DCloss, undersea_mask, network = [x[networksupport_mask] for x in (DClengths, DCloss, undersea_mask, network)]
network = np.array([n_node[node] for node in network.ravel()]).reshape(network.shape).astype(np.int64)

efficiency = 0.8 # round trip efficiency of storage
# efficiency = efficiency ** 0.5 # symmetric one-way efficiency

if scenario<=17:
    node = Nodel[scenario % 10]

    MLoad   = MLoad[:,   Nodel ==node]
    TSPV    = TSPV[:,    PVl   ==node]
    TSOnsW  = TSOnsW[:,  OnsWl ==node]
    TSOffW  = TSOffW[:,  OffWl==node]
    CHydro, CBio = [x[   Nodel ==node] for x in (CHydro, CBio)]

    Nodel_int, PVl_int, OnsWl_int, OffWl_int = [x[x==n_node[node]] for x in (Nodel_int, PVl_int, OnsWl_int, OffWl_int)]
    Nodel, PVl, OnsWl, OffWl                 = [x[x==node]         for x in (Nodel, PVl, OnsWl, OffWl)]
    network_mask = np.zeros(7, bool)
    network = np.empty((0,0), int)
    masked_DCloss = np.empty(0, float)

elif scenario>=21:
    coverage = [np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])][scenario % 10 - 1] 
    
    MLoad =  MLoad[:,  np.isin(nodesupportl, coverage)]
    TSPV =   TSPV[:,   np.isin(PVl,   coverage)]
    TSOnsW = TSOnsW[:, np.isin(OnsWl, coverage)]
    TSOffW = TSOffW[:, np.isin(OffWl, coverage)]
    CHydro, CBio = [x[ np.isin(Nodel, coverage)] for x in (CHydro, CBio)]
    
    if 'FNQ' not in coverage:
        MLoad[:, np.where(coverage=='QLD')[0][0]] /= 0.9
        
    coverage_int = np.array([n_node[node] for node in coverage])
    Nodel_int, PVl_int, OnsWl_int, OffWl_int = [x[np.isin(x, coverage_int)] for x in (Nodel_int, PVl_int, OnsWl_int, OffWl_int)]
    Nodel, PVl, OnsWl, OffWl                 = [x[np.isin(x, coverage)]     for x in (Nodel, PVl, OnsWl, OffWl)]

    network_mask = np.array([(network==j).sum(axis=1).astype(np.bool_) for j in Nodel_int]).sum(axis=0)==2
    network = network[network_mask,:]
    networkdict = {v:k for k, v in enumerate(Nodel_int)}
    #translate into indicies rather than Nodel_int values
    network = np.array([networkdict[n] for n in network.flatten()], np.int64).reshape(network.shape)
    masked_DCloss = DCloss[network_mask]

if 'WA' in Nodel or 'NT' in Nodel:
    raise NotImplementedError("Try a different scenario")

if scenario >= 31:
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    
    TSPV =   np.stack([TSPV[:,   PVl  ==node].mean(axis=1) for node in coverage]).T
    TSOnsW = np.stack([TSOnsW[:, OnsWl==node].mean(axis=1) for node in coverage]).T
    TSOffW = np.stack([TSOffW[:, OffWl==node].mean(axis=1) for node in coverage]).T
    # having full of zeros and setting lb,ub=0,0 makes code faster
    TSOffW = np.nan_to_num(TSOffW, False, 0)
    warnings.simplefilter('default', RuntimeWarning)
    
    Nodel_int, PVl_int, OnsWl_int, OffWl_int = [np.unique(x) for x in (Nodel_int, PVl_int, OnsWl_int, OffWl_int)]
    Nodel, PVl, OnsWl, OffWl                 = [np.unique(x) for x in (Nodel, PVl, OnsWl, OffWl)]
    
intervals, nodes = MLoad.shape
pzones, onswzones, offwzones = TSPV.shape[1], TSOnsW.shape[1],TSOffW.shape[1]

energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

firstyear = 2025
finalyear = firstyear+years-1


#%% 
# Find better way to sort these?
nhvdc = network_mask.sum()

pidx    =           pzones
onswidx = pidx    + onswzones
offwidx = onswidx + offwzones
spidx   = offwidx + nodes
seidx   = spidx   + nodes
hvidx   = seidx   + nhvdc
gidx    = hvidx   + nodes

MLoad = MLoad / 1000. #MW to GW
    
pv_zs_in_n =   [np.where(PVl  ==node)[0] + 1 for node in Nodel] # pyomo uses 1-indexing
onsw_zs_in_n = [np.where(OnsWl==node)[0] + 1 for node in Nodel] # pyomo uses 1-indexing
offw_zs_in_n = [np.where(OffWl==node)[0] + 1 for node in Nodel] # pyomo uses 1-indexing

pos_export_lines = [np.where(network[:,0]==n)[0] + 1 for n in range(nodes)] # pyomo uses 1-indexing
neg_export_lines = [np.where(network[:,1]==n)[0] + 1 for n in range(nodes)] # pyomo uses 1-indexing

npv    = len(PVl)
nonsw  = len(OnsWl)
noffw  = len(OffWl)

ndays = 365*years
intervals = int(ndays*24/resolution)

xlen = npv + nonsw + noffw + nodes*2 + nhvdc + nodes

costs = cost_factors(cost_source, DClengths, undersea_mask)
costs.hvdc = costs.hvdc[network_mask]

#%%
def zero_safe_divide(numerator, denominator, retval=0):
    return numerator / denominator if denominator != 0 else retval

class Solution:
    def __init__(self, model, years=years):
        self.scenario, self.nodes = scenario, nodes
        self.Nodel, self.PVl, self.OnsWl, self.OffWl = Nodel, PVl, OnsWl, OffWl
        
        self.network, self.network_mask, self.DCloss = network, network_mask, DCloss[network_mask]
        self.pos_export_lines = [np.where(network[:,0]==n)[0] for n in range(nodes)]
        self.neg_export_lines = [np.where(network[:,1]==n)[0] for n in range(nodes)]

        self.firstyear, self.years = firstyear, years
        self.finalyear = self.firstyear+self.years-1
        self.resolution = resolution
        self.intervals = int(years*365*24/resolution)
        
        self.StartCharge, self.efficiency = StartCharge, efficiency
        
        # capacities in GW and GWh
        self.cpv    = np.array([model.cpv[i].value   for i in model.cpv])
        self.consw  = np.array([model.consw[i].value for i in model.consw])
        self.coffw  = np.array([model.coffw[i].value for i in model.coffw])
        self.cgas   = np.array([model.cgas[i].value  for i in model.cgas])
        self.cphp   = np.array([model.cphp[i].value  for i in model.cphp])
        self.cphe   = np.array([model.cphe[i].value  for i in model.cphe]) 
        self.chvdc  = np.array([model.chvdc[i].value for i in model.chvdc])
        self.chydro = CHydro
        self.cbio   = CBio

        self.x = np.concatenate((self.cpv, self.consw, self.coffw, self.cphp, self.cphe, self.chvdc, self.cgas))

        self.cpv_n   = np.array([self.cpv[  np.where(self.PVl  ==node)[0]].sum() for node in self.Nodel])
        self.consw_n = np.array([self.consw[np.where(self.OnsWl==node)[0]].sum() for node in self.Nodel])
        self.coffw_n = np.array([self.coffw[np.where(self.OffWl==node)[0]].sum() for node in self.Nodel])

        # operations in MW and MWh
        self.Discharge = np.array([model.discharge[i].value for i in model.discharge]).reshape(-1, nodes) * 1000.  #GW to MW
        self.Charge    = np.array([model.charge[i].value    for i in model.charge   ]).reshape(-1, nodes) * 1000.
        self.Storage   = np.array([model.storage[i].value   for i in model.storage  ]).reshape(-1, nodes) * 1000.  
        
        self.Hydro     = np.array([model.hydro[i].value     for i in model.hydro    ]).reshape(-1, nodes) * 1000. 
        self.Bio       = np.array([model.bio[i].value       for i in model.bio      ]).reshape(-1, nodes) * 1000.
        self.Gas       = np.array([model.gas[i].value       for i in model.gas      ]).reshape(-1, nodes) * 1000.
        Hvdc_pos       = np.array([model.hvdc_pos[i].value  for i in model.hvdc_pos ]).reshape(-1, nhvdc) * 1000. 
        Hvdc_neg       = np.array([model.hvdc_neg[i].value  for i in model.hvdc_neg ]).reshape(-1, nhvdc) * 1000. 
        self.Hvdc = Hvdc_pos - Hvdc_neg

        self.Transmission = np.empty_like(self.Charge)
        for t in range(self.Charge.shape[0]):
            for n in range(self.Charge.shape[1]):
                self.Transmission[t, n]= (
                    + sum((Hvdc_neg[t, l]*(1-self.DCloss[l]) - Hvdc_pos[t, l] for l in self.pos_export_lines[n]))
                    + sum((Hvdc_pos[t, l]*(1-self.DCloss[l]) - Hvdc_neg[t, l]for l in self.neg_export_lines[n]))
                )
                
        self.Load =              MLoad[ :self.intervals, :] * 1000.
        self.PV   = self.cpv   * TSPV[  :self.intervals, :] * 1000.
        self.OnsW = self.consw * TSOnsW[:self.intervals, :] * 1000.
        self.OffW = self.coffw * TSOffW[:self.intervals, :] * 1000.
        self.PV   = np.stack([self.PV[:,   np.where(self.PVl  ==node)[0]].sum(axis=1) for node in self.Nodel]).T
        self.OnsW = np.stack([self.OnsW[:, np.where(self.OnsWl==node)[0]].sum(axis=1) for node in self.Nodel]).T
        self.OffW = np.stack([self.OffW[:, np.where(self.OffWl==node)[0]].sum(axis=1) for node in self.Nodel]).T
        
        self.Spillage = -np.minimum(0,
            self.Load 
            + self.Charge 
            - self.Discharge 
            - self.Hydro
            - self.Bio 
            - self.Gas
            - self.PV 
            - self.OnsW 
            - self.OffW
            - self.Transmission
            )
        
        self.GPV, self.GOnsW, self.GOffW, self.GHydro, self.GBio, self.GGas, self.GPHES = [
            x * self.resolution / self.years 
            for x in (self.PV.sum(axis=0), self.OnsW.sum(axis=0), self.OffW.sum(axis=0), self.Hydro.sum(axis=0), 
                      self.Bio.sum(axis=0), self.Gas.sum(axis=0), self.Discharge.sum(axis=0))] #TWh p.a.
        
        self.CFPV, self.CFOnsW, self.CFOffW, self.CFGas = (
            zero_safe_divide(G, c * 0.0876) for G, c in zip((self.GPV.sum(), self.GOnsW.sum(), self.GOffW.sum(), self.GGas.sum()), 
                                                            (self.cpv.sum(), self.consw.sum(), self.coffw.sum(), self.cgas.sum())))
        
        self.CostPV    = costs.pv     * self.cpv_n
        self.CostOnsW  = costs.onsw   * self.consw_n
        self.CostOffW  = costs.offw   * self.coffw_n
        self.CostGas   =(costs.gas[0] * self.cgas
                       + costs.gas[1] * self.GGas) #TWh p.a. to MWh p.a.)
        self.CostHydro = costs.hydro * self.GHydro
        self.CostBio   = (costs.hydro + 0.1) * self.GBio 
        self.CostPH = (costs.phes[0] * self.cphp +
                       costs.phes[1] * self.cphe +
                       costs.phes[2] * self.GPHES + # ignore vom
                       costs.phes[3])
        self.CostDC    = costs.hvdc * self.chvdc
        self.CostAC    = (self.cpv_n + self.consw_n + self.coffw_n + self.cgas)*costs.ac

        self.Energy = self.Load.sum() * self.resolution / self.years  # MWh p.a.

        self.LCOE = (self.CostPV.sum() + self.CostOnsW.sum() + self.CostOffW.sum() + 
                     self.CostGas.sum() + self.CostHydro.sum() + self.CostBio.sum() + 
                     self.CostPH.sum() + self.CostDC.sum() + self.CostAC.sum()) / self.Energy
        self.LCOG = ((self.CostPV.sum() + self.CostOnsW.sum() + self.CostOffW.sum() + 
                      self.CostHydro.sum() + self.CostBio.sum() + self.CostGas.sum()) /
                     (self.GPV.sum() + self.GOnsW.sum() + self.GOffW.sum() + self.GHydro.sum() + self.GBio.sum() + self.GGas.sum()))
        
        
        self.LCOGP    = zero_safe_divide(self.CostPV.sum()    , self.GPV.sum()   )
        self.LCOGOnsW = zero_safe_divide(self.CostOnsW.sum()  , self.GOnsW.sum() )
        self.LCOGOffW = zero_safe_divide(self.CostOffW.sum()  , self.GOffW.sum() )
        self.LCOGH    = zero_safe_divide(self.CostHydro.sum() , self.GHydro.sum())
        self.LCOGB    = zero_safe_divide(self.CostBio.sum()   , self.GBio.sum()  )
        self.LCOGG    = zero_safe_divide(self.CostGas.sum()   , self.GGas.sum()  )

        self.LCOB = self.LCOE - self.LCOG
        self.LCOBS = self.CostPH.sum() / self.Energy
        self.LCOBT = (self.CostDC.sum() + self.CostAC.sum()) / self.Energy
        self.LCOBL = self.LCOB - self.LCOBS - self.LCOBT


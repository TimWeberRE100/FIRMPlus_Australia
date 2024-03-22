# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from Optimisation import scenario
from numba import jit, float64, int32, types, int64
from numba.experimental import jitclass

Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
PVl =   np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)


_, Nodel_int = np.unique(Nodel, return_inverse=True)
_, PVl_int = np.unique(Nodel, return_inverse=True)
_, Windl_int = np.unique(Nodel, return_inverse=True)

Nodel_int, PVl_int, Windl_int = (x.astype(np.int32) for x in (Nodel_int, PVl_int, Windl_int))

resolution = 0.5
firstyear, finalyear, timestep = (2020, 2029, 1)

MLoad = np.genfromtxt('Data/electricity.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel))) # EOLoad(t, j), MW

TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) # TSPV(t, i), MW
TSWind = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Windl))) # TSWind(t, i), MW

assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(float)
CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW
CBaseload = np.array([0, 0, 0, 0, 0, 1.0, 0, 0]) # 24/7, GW
CPeak = CHydro + CBio - CBaseload # GW

# FQ, NQ, NS, NV, AS, SW, only TV constrained
DCloss = np.array([1500, 1000, 1000, 800, 1200, 2400, 400]) * 0.03 * pow(10, -3)
CDC6max = 3 * 0.63 # GW


efficiency = 0.8
factor = np.genfromtxt('Data/factor.csv', delimiter=',', usecols=1)


if scenario<=17:
    node = Nodel[scenario % 10]

    MLoad = MLoad[:, np.where(Nodel==node)[0]]
    TSPV = TSPV[:, np.where(PVl==node)[0]]
    TSWind = TSWind[:, np.where(Windl==node)[0]]
    CHydro, CBio, CBaseload, CPeak = [x[np.where(Nodel==node)[0]] for x in (CHydro, CBio, CBaseload, CPeak)]

    Nodel_int, PVl_int, Windl_int = [x[np.where(Nodel==node)[0]] for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [x[np.where(x==node)[0]] for x in (Nodel, PVl, Windl)]

elif scenario>=21:
    coverage = [np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])][scenario % 10 - 1] 
    
    MLoad = MLoad[:, np.where(np.in1d(Nodel, coverage)==True)[0]]
    TSPV = TSPV[:, np.where(np.in1d(PVl, coverage)==True)[0]]
    TSWind = TSWind[:, np.where(np.in1d(Windl, coverage)==True)[0]]
    CHydro, CBio, CBaseload, CPeak = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (CHydro, CBio, CBaseload, CPeak)]
    
    if 'FNQ' not in coverage:
        MLoad[:, np.where(coverage=='QLD')[0][0]] /= 0.9
    
    Nodel_int, PVl_int, Windl_int = [x[np.where(np.in1d(Nodel, coverage)==True)[0]] for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [x[np.where(np.in1d(x, coverage)==True)[0]] for x in (Nodel, PVl, Windl)]

    
intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)
pzones, wzones = (TSPV.shape[1], TSWind.shape[1])
pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes)

energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

GBaseload = np.tile(CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW

lb = np.array([0.]  * pzones + [0.]   * wzones + contingency   + [0.])
ub = np.array([50.] * pzones + [50.]  * wzones + [50.] * nodes + [5000.])

#%%
from Simulation import Reliability 
from Network import Transmission

@jit(nopython=True)
def F(S):
    nvec = S.nvec
    
    Deficit = Reliability(S, flexible=np.zeros((intervals, 1) , dtype=np.float64)) # Sj-EDE(t, j), MW
    Flexible = Deficit.sum(axis=0) * resolution / years / efficiency # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
    PenHydro = np.maximum(0, Hydro - 20 * 1000000) # TWh p.a. to MWh p.a.

    Deficit = Reliability(S, flexible=np.ones((intervals, 1), dtype=np.float64)*CPeak.sum()*1000) # Sj-EDE(t, j), GW to MW
    PenDeficit = np.maximum(0, Deficit.sum(axis=0) * resolution) # MWh

    TDC_abs = Transmission(S) if scenario>=21 else np.zeros((intervals, len(DCloss)), dtype=np.float64)  # TDC: TDC(t, k), MW
    TDC_abs = np.atleast_3d(np.abs(TDC_abs))

    CDC = np.zeros((len(DCloss), nvec), dtype=np.float64)
    for i in range(intervals):
        for j in range(len(DCloss)):
            CDC[j, :] = np.maximum(TDC_abs[i, j, :], CDC[j,:])
    CDC = CDC * 0.001 # CDC(k), MW to GW
    PenDC = np.maximum(0, CDC[6,:] - CDC6max) * 0.001 # GW to MW

    # numba is fussy about generation of tuples and about stacking arrays of different dimensions
    costitems = np.vstack((S.CPV.sum(axis=0), S.CWind.sum(axis=0), S.CPHP.sum(axis=0), S.CPHS,
                            S.CPV.sum(axis=0), S.CWind.sum(axis=0), Hydro * 0.000001,
                            np.repeat(-1.0, nvec), np.repeat(-1.0, nvec),))
    costitems = np.vstack((costitems, CDC))
    reindex = np.concatenate((np.arange(4), np.arange(9, 16), np.arange(4, 9)))

    cost = factor.reshape(-1,1) * costitems[reindex]
    cost = cost.sum(axis=0)

    loss = TDC_abs.sum(axis=0) * DCloss.reshape(-1,1)
    loss = loss.sum(axis=0) * 0.000000001 * resolution / years # PWh p.a.
    LCOE = cost / np.abs(energy - loss)
    
    return LCOE, (PenHydro+PenDeficit+PenDC)


# Specify the types for jitclass
solution_spec = [
    ('x', float64[:,:]),  # x is 2d array
    ('nvec', int32),
    ('MLoad', float64[:, :, :]),  # 3D array of floats
    ('intervals', int32),
    ('nodes', int32),
    ('resolution',float64),
    ('CPV', float64[:, :]), # 2D array of floats
    ('CWind', float64[:, :]), # 2D array of floats
    ('GPV', float64[:, :, :]),  # 3D array of floats
    ('GWind', float64[:, :, :]),  # 3D array of floats
    ('CPHP', float64[:, :]),
    ('CPHS', float64[:]),
    ('efficiency', float64),
    # ('CInter', float64[:]),
    # ('GInter', float64[:, :]),  # 2D array of floats
    # ('Interl_int', int32[:]),
    # ('node', types.unicode_type),
    ('Nodel_int', int32[:]), 
    ('PVl_int', int32[:]),
    ('Windl_int', int32[:]),
    ('GBaseload', float64[:, :, :]),  # 3D array of floats
    ('CPeak', float64[:]),  # 1D array of floats
    ('CHydro', float64[:]),  # 1D array of floats
    ('EHydro', float64[:]),  # 1D array of floats
    ('allowance', float64),
    ('flexible', float64[:,:]),
    ('Discharge', float64[:,:]),
    ('Charge', float64[:,:]),
    ('Storage', float64[:,:]),
    ('Deficit', float64[:,:]),
    ('Spillage', float64[:,:]),
    ('Penalties', float64[:]),
    ('Lcoe', float64[:]),
]

@jitclass(solution_spec)
class Solution:
    #A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)
    
    def __init__(self, x):
        # input vector should have shape (sidx+1, n) i.e. vertical input vectors
        self.x = x
        self.nvec = x.shape[1]
        
        self.intervals, self.nodes = intervals, nodes
        self.resolution = resolution
       
        shape3d = (intervals, nodes, 1)
        self.MLoad = MLoad.reshape(shape3d)

        self.CPV = x[: pidx, :]  # CPV(i), GW
        self.CWind = x[pidx: widx, :]  # CWind(i), GW
        
        # Manually replicating np.tile functionality for CPV and CWind
        CPV_tiled = np.zeros((intervals, *self.CPV.shape))
        CWind_tiled = np.zeros((intervals, *self.CWind.shape))
        # CInter_tiled = np.zeros((intervals, len(self.CWind)))
        for i in range(intervals):
            for j in range(len(self.CPV)):
                CPV_tiled[i, j, :] = self.CPV[j, :]
            for j in range(len(self.CWind)):
                CWind_tiled[i, j, :] = self.CWind[j, :]
                
        self.GPV = TSPV.reshape((*TSPV.shape, 1)) * CPV_tiled * 1000.  # GPV(i, t), GW to MW
        self.GWind = TSWind.reshape((*TSWind.shape, 1)) * CWind_tiled * 1000.  # GWind(i, t), GW to MW

        self.CPHP = x[widx: sidx, :]  # CPHP(j), GW
        self.CPHS = x[sidx, :]  # S-CPHS(j), GWh
        self.efficiency = efficiency

        self.Nodel_int, self.PVl_int, self.Windl_int = Nodel_int, PVl_int, Windl_int
        # self.node = node

        self.GBaseload = GBaseload.reshape(shape3d)
        self.CPeak = CPeak
        self.CHydro = CHydro
        #self.EHydro = EHydro

        self.Lcoe, self.Penalties = F(self)


    # # Not currently supported by jitclass
    # def __repr__(self):
    #     """S = Solution(list(np.ones(64))) >> print(S)"""
    #     return 'Solution({})'.format(self.x)

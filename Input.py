# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from Optimisation import scenario, node
from numba import float64, int32, types, int64
from numba.experimental import jitclass

Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
PVl =   np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)

_, Nodel_int = np.unique(Nodel, return_inverse=True)
_, PVl_int = np.unique(Nodel, return_inverse=True)
_, Windl_int = np.unique(Nodel, return_inverse=True)

Nodel_int = Nodel_int.astype(np.int32)
PVl_int = PVl_int.astype(np.int32)
Windl_int = Windl_int.astype(np.int32)

resolution = 0.5

MLoad = np.genfromtxt('Data/electricity.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel))) # EOLoad(t, j), MW

TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) # TSPV(t, i), MW
TSWind = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Windl))) # TSWind(t, i), MW

assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(np.float)
CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW
CBaseload = np.array([0, 0, 0, 0, 0, 1.0, 0, 0]) # 24/7, GW
CPeak = CHydro + CBio - CBaseload # GW

# FQ, NQ, NS, NV, AS, SW, only TV constrained
DCloss = np.array([1500, 1000, 1000, 800, 1200, 2400, 400]) * 0.03 * pow(10, -3)

efficiency = 0.8
factor = np.genfromtxt('Data/factor.csv', delimiter=',', usecols=1)

firstyear, finalyear, timestep = (2020, 2029, 1)

if scenario<=17:
    node = Nodel[scenario % 10]

    MLoad = MLoad[:, np.where(Nodel==node)[0]]
    TSPV = TSPV[:, np.where(PVl==node)[0]]
    TSWind = TSWind[:, np.where(Windl==node)[0]]
    CHydro, CBio, CBaseload, CPeak = [x[np.where(Nodel==node)[0]] for x in (CHydro, CBio, CBaseload, CPeak)]

    Nodel_int, PVl_int, Windl_int = [x[np.where(Nodel==node)[0]] for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [x[np.where(x==node)[0]] for x in (Nodel, PVl, Windl)]
    
intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)
pzones, wzones = (TSPV.shape[1], TSWind.shape[1])
pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes)

energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

GBaseload = np.tile(CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW

# Specify the types for jitclass
solution_spec = [
    ('x', float64[:]),  # Assuming x is a list of floats
    ('MLoad', float64[:, :]),  # 2D array of floats
    ('intervals', int32),
    ('nodes', int32),
    ('resolution',float64),
    ('CPV', float64[:]),
    ('CWind', float64[:]),
    ('GPV', float64[:, :]),  # 2D array of floats
    ('GWind', float64[:, :]),  # 2D array of floats
    ('CPHP', float64[:]),
    ('CPHS', float64),
    ('efficiency', float64),
    ('CInter', float64[:]),
    ('GInter', float64[:, :]),  # 2D array of floats
    ('Nodel_int', int32[:]), 
    ('PVl_int', int32[:]),
    ('Windl_int', int32[:]),
    ('Interl_int', int32[:]),
    ('node', types.unicode_type),
    ('GBaseload', float64[:, :]),  # 2D array of floats
    ('CPeak', float64[:]),  # 1D array of floats
    ('CHydro', float64[:]),  # 1D array of floats
    ('EHydro', float64[:]),  # 1D array of floats
    ('allowance', float64),
    ('flexible', float64[:,:]),
    ('Discharge', float64[:,:]),
    ('Charge', float64[:,:]),
    ('Storage', float64[:,:]),
    ('Deficit', float64[:,:]),
    ('Spillage', float64[:,:])
]

@jitclass(solution_spec)
class Solution:
    #A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)
    
    def __init__(self, x):
        self.x = x
        self.MLoad = MLoad
        self.intervals = intervals
        self.nodes = nodes
        self.resolution = resolution

        self.CPV = x[: pidx]  # CPV(i), GW
        self.CWind = x[pidx: widx]  # CWind(i), GW
        """ if node == 'Super2':
            self.CInter = NumbaList(x[sidx+1: iidx]) # CInter(j), GW
        else:
            self.CInter = NumbaList([0.0])  # CInter(j), GW """
        
        # Manually replicating np.tile functionality for CPV and CWind
        CPV_tiled = np.zeros((intervals, len(self.CPV)))
        CWind_tiled = np.zeros((intervals, len(self.CWind)))
        #CInter_tiled = np.zeros((intervals, len(self.CWind)))
        for i in range(intervals):
            for j in range(len(self.CPV)):
                CPV_tiled[i, j] = self.CPV[j]
            for j in range(len(self.CWind)):
                CWind_tiled[i, j] = self.CWind[j]
            """ for j in range(len(self.CInter)):
                CInter_tiled[i, j] = self.CInter[j] """

        self.GPV = TSPV * CPV_tiled * 1000  # GPV(i, t), GW to MW
        self.GWind = TSWind * CWind_tiled * 1000  # GWind(i, t), GW to MW
        #self.GInter = CWind_tiled * 1000  # GInter(j, t), GW to MW

        self.CPHP = x[widx: sidx]  # CPHP(j), GW
        self.CPHS = x[sidx]  # S-CPHS(j), GWh
        self.efficiency = efficiency

        self.Nodel_int = Nodel_int
        self.PVl_int = PVl_int
        self.Windl_int = Windl_int
        #self.Interl = NumbaList([str(item) for item in Interl])
        self.node = node

        self.GBaseload = GBaseload
        self.CPeak = CPeak
        self.CHydro = CHydro
        #self.EHydro = EHydro

    def __repr__(self):
        """S = Solution(list(np.ones(64))) >> print(S)"""
        return 'Solution({})'.format(self.x)


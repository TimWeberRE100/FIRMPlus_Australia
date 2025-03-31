# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit, float64, int64, prange, boolean
from numba.experimental import jitclass

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', default=1000, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=100, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')
parser.add_argument('-s', default=21, type=int, required=False, help='11, 12, 13, ...')
parser.add_argument('-cb', default=0, type=int, required=False, help='Callback: 0-None, 1-generation elites, 2-everything')
parser.add_argument('-ver', default=1, type=int, required=False, help='Boolean - print progress to console')
parser.add_argument('-w', default=-1, type=int, required=False, help='Maximum number of cores to parallelise over')
parser.add_argument('-y', default=1, type=int, required=False, help='No. of years to simulate')

args = parser.parse_args()
assert args.w > 0 or args.w in (-1, -2)
scenario = args.s

from Costs import Raw_Costs
from Simulation import Simulate

Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
PVl =   np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)

n_node = dict((name, i) for i, name in enumerate(Nodel))
Nodel_int, PVl_int, Windl_int = (np.array([n_node[node] for node in x], dtype=np.int64) for x in (Nodel, PVl, Windl))

MLoad = np.genfromtxt('Data/electricity.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel))) # EOLoad(t, j), MW
MLoad /= 1000 # MW/GW

resolution = 0.5
years = int(resolution * len(MLoad) / 8760) if args.y==-1 else args.y
intervals = int(years*8760/resolution)
firstyear, finalyear, timestep = (2020, 2020+args.y-1, 1)

TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) # TSPV(t, i), MW
TSWind = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Windl))) # TSWind(t, i), MW

assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(float)
CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW
CBaseload = np.array([0, 0, 0, 0, 0, 1.0, 0, 0]) # 24/7, GW
CPeak = CHydro + CBio - CBaseload # GW

# FQ, NQ, NS, NV, AS, SW, only TV constrained
Lengths = np.array([1500, 1000, 1000, 800, 1200, 2400, 400], dtype=np.int64)
DCloss = Lengths * 0.03 * 0.001 #3% per 1000 km
undersea_mask = np.array([0, 0, 0, 0, 0, 0, 1], dtype=bool)

efficiency = 0.8
factor = np.genfromtxt('Data/factor.csv', delimiter=',', usecols=1)

if scenario<=17:
    node = Nodel[scenario % 10]

    MLoad = MLoad[:, Nodel==node]
    TSPV = TSPV[:, PVl==node]
    TSWind = TSWind[:, Windl==node]
    CHydro, CBio, CBaseload, CPeak = [x[Nodel==node] for x in (CHydro, CBio, CBaseload, CPeak)]

    Nodel_int, PVl_int, Windl_int = [x[x==n_node[node]] for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [x[x==node] for x in (Nodel, PVl, Windl)]
    network = np.empty((0,0,0,0), dtype=np.int64)
    network_mask = np.zeros(7, dtype=np.bool_)
    directconns = np.empty((0,0), dtype=np.int64) 
    trans_mask = np.empty((0,0), dtype=np.bool_)

elif scenario>=21:
    coverage = [np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])][scenario % 10 - 1] 
    
    network = np.array([[0, 3], #FNQ-QLD
                        [1, 3], #NSW-QLD
                        [1, 4], #NSW-SA
                        [1, 6], #NSW-VIC
                        [2, 4], #NT-SA
                        [4, 7], #SA-WA
                        [5, 6], #TAS-VIC
                        ], dtype=np.int64)
    
    MLoad = MLoad[:, np.isin(Nodel, coverage)]
    TSPV = TSPV[:, np.isin(PVl, coverage)]
    TSWind = TSWind[:, np.isin(Windl, coverage)]
    CHydro, CBio, CBaseload, CPeak = [x[np.isin(Nodel, coverage)] for x in (CHydro, CBio, CBaseload, CPeak)]
    
    if 'FNQ' not in coverage:
        MLoad[:, np.where(coverage=='QLD')[0][0]] /= 0.9
    
    coverage_int = np.array([n_node[node] for node in coverage])
    
    Nodel_int, PVl_int, Windl_int = [x[np.isin(x, coverage_int)] for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [x[np.isin(x, coverage)] for x in (Nodel, PVl, Windl)]
    
    from Network import generate_network 
    network, network_mask, trans_mask, directconns, triangulars = generate_network(network, Nodel_int)

    
MLoad, TSPV, TSWind = (x[:intervals, :] for x in (MLoad, TSPV, TSWind))

nhvdc = network_mask.sum()   
nodes = MLoad.shape[1]

pzones, wzones = (TSPV.shape[1], TSWind.shape[1])
pidx, widx = pzones, pzones + wzones
spidx, seidx = pzones + wzones + nodes, pzones + wzones + nodes + nodes


energy = MLoad.sum() * 1000 * resolution / years # MWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

lb = np.array([0.]  * pzones + [0.]   * wzones + [0.] * nodes  + [0.] * nodes   + [0.] * nhvdc)
ub = np.array([32.] * pzones + [32.]  * wzones + [32.] * nodes + [500.] * nodes + [100.]* nhvdc)
              # list(np.array(CDCmax)[network_mask]))

x0 = np.concatenate((
    MLoad.sum()/intervals*0.75 / len(PVl) / TSPV.mean(axis=0), 
    MLoad.sum()/intervals*0.75 / len(Windl) / TSWind.mean(axis=0), 
    MLoad.max(axis=0)*1.1, 
    MLoad.max(axis=0)*36, 
    np.repeat(MLoad.max()*0.75, nhvdc)))
x0 = np.minimum(ub, x0)

triangulars = np.array([0,1,3,6,10,15,21,28], np.int64)


#%%
@njit
def zero_safe_division(numerator, denominator, error=0):
    return error if denominator == 0 else numerator/denominator

@njit()
def Evaluate(S, cost_model):
    S._instantiate_operations()
    Simulate(S)

    S.Penalties = np.maximum(0, S.MDeficit.sum()) # GWh/resolution

    CHVI = np.zeros(len(network_mask), dtype=np.float64)
    CHVI[network_mask] = S.CHVI

    cost = np.array([
        # generation capex 
        S.CPV.sum()   * cost_model.pv[0], 
        S.CWind.sum() * cost_model.onsw[0], 
        0, # S.CGas.sum()  * cost_model.gas[0],
        (S.CHydro.sum()+S.CBio.sum()+S.CBaseload.sum()) * cost_model.hydro[0],
        
        # generation fom
        S.CPV.sum()   * cost_model.pv[1], 
        S.CWind.sum() * cost_model.onsw[1], 
        0, # S.CGas.sum()  * cost_model.gas[1],
        (S.CHydro.sum()+S.CBio.sum()+S.CBaseload.sum()) * cost_model.hydro[1],
        
        # generation vom
        # pv, onsw, battery are 0
        0, # S.GGas.sum() * S.resolution / S.years * cost_model.gas[2],
        (S.MFlexible.sum() + S.CBaseload.sum()*S.intervals
         ) * S.resolution / S.years * cost_model.hydro[2],
        
        # storage 
        S.CPHP.sum() * cost_model.phes[0],
        S.CPHS.sum() * cost_model.phes[1],
        S.CPHP.sum() * cost_model.phes[2],
        S.MDischarge.sum() * S.resolution / S.years * cost_model.phes[3], 
        cost_model.phes[4],
        ] +
        
        # transmission network
        list((S.CPV.sum() + S.CWind.sum() + 
              # S.CGas.sum() + 
              S.CHydro.sum() + S.CBio.sum())*cost_model.ac) +
        list((CHVI * cost_model.hvi).sum(axis=1))
        ) 
        
    # Levelised Costs of:
    # Electricity
    S.LCOE = cost.sum() / energy
    # Generation
    S.LCOG = cost[:10].sum() / (1000*S.resolution/S.years*(S.MPV.sum()+S.MWind.sum()
                                                           # +S.MGas.sum()
                                +S.MFlexible.sum()+S.CBaseload.sum()*S.intervals))
    # Storage
    # S.LCOSP = zero_safe_division(cost[10:15].sum(), S.MDischarge.sum()*S.resolution/S.years)
    # Balancing - Storage
    S.LCOBS = cost[10:15].sum()/energy
    # Balancing - Transmission 
    S.LCOBT = cost[15:].sum()/energy
    # Balancing - Spillage
    S.LCOBL = S.LCOE - S.LCOG - S.LCOBS - S.LCOBT
    
    S.CAPEX = sum([cost[i] for i in [0,1,2,3,10,11,14,15,16,17,18,19,20,21]])/energy
    S.OPEX = S.LCOE - S.CAPEX
    
    return S.LCOE, S.Penalties


# Specify the types for jitclass
solution_spec = [
    ('x', float64[:]),
    ('intervals', int64),
    ('nodes', int64),
    ('nhvdc', int64),
    ('resolution',float64),
    ('years',int64),
    ('efficiency', float64),
    ('Flex_res', float64),
    ('Nodel_int', int64[:]), 
    # ('PVl_int', int64[:]),
    # ('Windl_int', int64[:]),
    ('networksteps', int64),
    ('network_mask', boolean[:]),
    ('network', int64[:, :, :, :]),
    ('directconns', int64[:,:]),

    # Capacities in GW/GWh
    ('CPV', float64[:]),
    ('CWind', float64[:]),
    ('CPHP', float64[:]),
    ('CPHS', float64[:]),
    ('CHVI', float64[:]),
    ('CBaseload', float64[:]),
    ('CPeak', float64[:]),
    ('CHydro', float64[:]),
    ('CBio', float64[:]),

    # Nodally diaggregated operations in GW/GWh
    ('MFlexible', float64[:,:]),
    ('MDischarge', float64[:, :]),
    ('MCharge', float64[:, :]),
    ('MStorage', float64[:, :]),
    ('MDeficit', float64[:, :]),
    ('MSpillage', float64[:, :]),
    ('MNetload' ,float64[:, :]),
    ('MImport', float64[:, :]),
    ('MExport', float64[:, :]),
    ('MPV', float64[:, :]),
    ('MWind', float64[:, :]),
    ('MLoad', float64[:, :]),
    ('MBaseload', float64[:, :]),
    ('MHydro', float64[:, :]),
    ('MBio', float64[:, :]),

    ('TDC', float64[:, :]),
    ('Topology', float64[:, :]),
    ('trans_mask', boolean[:,:]),
    ('TImport', float64[:,:,:]),
    ('TExport', float64[:,:,:]),

    ('Penalties', float64),
    ('LCOE', float64),
    ('LCOG', float64),
    ('LCOSP', float64),
    ('LCOSB', float64),
    ('LCOBS', float64),
    ('LCOBT', float64),
    ('LCOBL', float64),
    ('CAPEX', float64),
    ('OPEX', float64),

]

@jitclass(solution_spec)
class Solution:
    #A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)
    def __init__(self, x):
        assert len(x) == len(lb)
        
        self.x = x

        self.Flex_res = 20000 /resolution*years
        self.intervals, self.nodes = intervals, nodes
        self.nhvdc = network_mask.sum()
        self.resolution, self.efficiency = resolution, efficiency
        self.years = years
        self.network, self.directconns = network, directconns
        self.networksteps = np.where(triangulars == network.shape[2])[0][0]

        self.Nodel_int = Nodel_int
        # self.PVl_int, self.Windl_int = PVl_int, Windl_int

        self.trans_mask = trans_mask
       
        self.CPV = x[: pidx]  # CPV(i), GW
        self.CWind = x[pidx: widx]  # CWind(i), GW
        self.CPHP = x[widx: spidx]  # CPHP(j), GW
        self.CPHS = x[spidx: seidx]  # S-CPHS(j), GWh
        self.CHVI = x[seidx:]
        self.CBaseload = CBaseload
        self.CPeak = CPeak
        self.CHydro = CHydro
        self.CBio = CBio
        
    def _instantiate_operations(self):
        self.MLoad = MLoad

        self.MPV, self.MWind = np.zeros((intervals, nodes)), np.zeros((intervals, nodes))
        for i, n in enumerate(Nodel_int):
            self.MPV[:, i] += (TSPV[:, PVl_int==n] * self.CPV[PVl_int==n]).sum(axis=1)
            self.MWind[:, i] += (TSWind[:, Windl_int==n] * self.CWind[Windl_int==n]).sum(axis=1)
        
        self.MNetload = self.MLoad - self.MPV - self.MWind - self.CBaseload
        self.MDeficit = np.maximum(0, self.MNetload)
        self.MSpillage = -np.minimum(0, self.MNetload)
        
        self.MFlexible = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        
        self.MDischarge = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        self.MCharge    = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        self.MStorage   = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        self.MStorage[-1] = 0.5*self.CPHS

        self.TImport = np.zeros((self.intervals, self.nhvdc, self.nodes), dtype=np.float64)
        self.TExport = np.zeros((self.intervals, self.nhvdc, self.nodes), dtype=np.float64)
        self.TDC = np.zeros((self.intervals, self.nodes), dtype=np.float64)


if __name__=='__main__':
    cost_model = Raw_Costs(scenario, Lengths, undersea_mask, network_mask).CostFactors()

    # x = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
    solution = Solution(x0)
    Evaluate(solution, cost_model)
    print(solution.LCOE, solution.Penalties)
    
    @njit
    def test(cost_model, disp=False):
        x = (ub-lb)*np.random.rand(len(lb))
        solution = Solution(x)
        Evaluate(solution, cost_model)
        if disp:
            print(solution.LCOE, solution.Penalties)
    test(cost_model)
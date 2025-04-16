
import numpy as np
from numba import boolean, float64, int64, njit, types, objmode  # type: ignore
from numba.experimental import jitclass  # type: ignore
from numba.typed.typeddict import Dict as TypedDict

from firm.Simulation import Simulate
from firm.Utils import zero_safe_division, array_max, cclock
from firm.Network import generate_network


Nodel = np.array(["FNQ", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"])
PVl = np.array(
    ["NSW"] * 7
    + ["FNQ"] * 1
    + ["QLD"] * 2
    + ["FNQ"] * 3
    + ["SA"] * 6
    + ["TAS"] * 0
    + ["VIC"] * 1
    + ["WA"] * 1
    + ["NT"] * 1
)
Windl = np.array(
    ["NSW"] * 8
    + ["FNQ"] * 1
    + ["QLD"] * 2
    + ["FNQ"] * 2
    + ["SA"] * 8
    + ["TAS"] * 4
    + ["VIC"] * 4
    + ["WA"] * 3
    + ["NT"] * 1
)

n_node = dict((name, i) for i, name in enumerate(Nodel))
Nodel_int, PVl_int, Windl_int = (np.array([n_node[node] for node in x], dtype=np.int64) for x in (Nodel, PVl, Windl))

MLoad = np.genfromtxt("Data/electricity.csv", delimiter=",", skip_header=1, usecols=range(4, 4 + len(Nodel))) 
MLoad /= 1000  # MW to GW

TSPV = np.genfromtxt("Data/pv.csv", delimiter=",", skip_header=1, usecols=range(4, 4 + len(PVl)))
TSWind = np.genfromtxt("Data/wind.csv", delimiter=",", skip_header=1, usecols=range(4, 4 + len(Windl)))

assets = np.genfromtxt("Data/hydrobio.csv", dtype=None, delimiter=",", encoding=None)[1:, 1:].astype(float)
CHydro, CBio = (assets[:, x] * 0.001 for x in range(assets.shape[1])) # MW to GW
CBaseload = np.array([0, 0, 0, 0, 0, 1.0, 0, 0])  # 24/7, GW
CPeak = CHydro + CBio - CBaseload  # GW

# FQ, NQ, NS, NV, AS, SW, only TV constrained
lengths = np.array([1500, 1000, 1000, 800, 1200, 2400, 400], dtype=np.int64)
DCloss = lengths * 0.03 * 0.001  # 3% per 1000 km
undersea_mask = np.array([0, 0, 0, 0, 0, 0, 1], dtype=bool)

coverage = [
    np.array(["NSW", "QLD", "SA", "TAS", "VIC"]),
    np.array(["NSW", "QLD", "SA", "TAS", "VIC", "WA"]),
    np.array(["NSW", "NT", "QLD", "SA", "TAS", "VIC"]),
    np.array(["NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"]),
    np.array(["FNQ", "NSW", "QLD", "SA", "TAS", "VIC"]),
    np.array(["FNQ", "NSW", "QLD", "SA", "TAS", "VIC", "WA"]),
    np.array(["FNQ", "NSW", "NT", "QLD", "SA", "TAS", "VIC"]),
    np.array(["FNQ", "NSW", "NT", "QLD", "SA", "TAS", "VIC", "WA"]),
]
coverage_int = [np.array([n_node[node] for node in node_array], dtype=np.int64) for node_array in coverage]
coverage_maxlen = max((len(c) for c in coverage))
coverage_int = np.stack([np.pad(c, (0, coverage_maxlen - len(c)), constant_values=-1) for c in coverage_int])


basic_network = np.array(
    [
        [0, 3],  # FNQ-QLD
        [1, 3],  # NSW-QLD
        [1, 4],  # NSW-SA
        [1, 6],  # NSW-VIC
        [2, 4],  # NT-SA
        [4, 7],  # SA-WA
        [5, 6],  # TAS-VIC
    ],
    dtype=np.int64,
)

data_spec=[
    ("scenario", int64),
    ("profiling", boolean),
    ("resolution", float64),
    ("efficiency", float64),
    ("years", int64),
    ("intervals", int64),
    ("MLoad", float64[:, :]),
    ("TSPV", float64[:, :]),
    ("TSWind", float64[:, :]),
    ("CHydro", float64[:]),
    ("CBio", float64[:]),
    ("CBaseload", float64[:]),
    ("CPeak", float64[:]),
    ("coverage_int", int64[:]),
    ("Nodel_int", int64[:]),
    ("PVl_int", int64[:]),
    ("Windl_int", int64[:]),
    ("basic_network", int64[:, :]),
    ("network", int64[:, :, :, :]),
    ("network_mask", boolean[:]),
    ("directconns", int64[:, :]),
    ("trans_mask", boolean[:, :]),
    ("triangulars", int64[:]),
    ("nhvi", int64),
    ("nodes", int64),
    ("pzones", int64),
    ("wzones", int64),
    ("pidx", int64),
    ("widx", int64),
    ("spidx", int64),
    ("seidx", int64),
    ("energy", float64),
    ("lb", float64[:]),
    ("ub", float64[:]),
    ("x0", float64[:]),
    ]

@jitclass(data_spec)
class Solution_data:
    def __init__(
            self, 
            scenario: int, 
            years: int,
            profiling: bool,
            ):
        self.scenario = scenario
        self.profiling = profiling
        self.resolution = 0.5
        self.efficiency = 0.8
        
        maxyears = int(self.resolution * len(MLoad) / 8760) 
        if years == -1:
            self.years = maxyears
        elif years <= maxyears:
            self.years = years
        else: 
            raise Exception
        self.intervals = int(self.years * 8760 / self.resolution)
        
        if scenario <= 17:
            node = Nodel_int[scenario % 10]
        
            self.MLoad =  np.atleast_2d(MLoad[: self.intervals,  Nodel_int == node]).T
            self.TSPV =   np.atleast_2d(TSPV[: self.intervals,   PVl_int ==   node]).T
            self.TSWind = np.atleast_2d(TSWind[: self.intervals, Windl_int == node]).T
            
            self.CHydro =    CHydro[   Nodel_int == node]
            self.CBio =      CBio[     Nodel_int == node]
            self.CBaseload = CBaseload[Nodel_int == node]
            self.CPeak =     CPeak[    Nodel_int == node]
        
            self.Nodel_int = Nodel_int[Nodel_int == node]
            self.PVl_int =   PVl_int[  PVl_int ==   node]
            self.Windl_int = Windl_int[Windl_int == node]
            # Nodel, PVl, Windl = [x[x == node] for x in (Nodel, PVl, Windl)]
            self.basic_network=np.empty((0,0), np.int64)
            self.network = np.empty((0, 0, 0, 0), dtype=np.int64)
            self.network_mask = np.zeros(len(basic_network), dtype=np.bool_)
            self.directconns = np.empty((0, 0), dtype=np.int64)
            self.trans_mask = np.empty((0, 0), dtype=np.bool_)
            self.triangulars = np.zeros(1, np.int64)
        
        elif scenario >= 21:
            self.coverage_int = coverage_int[self.scenario % 10 - 1]
            self.coverage_int = self.coverage_int[self.coverage_int != -1]
        
            self.MLoad =  MLoad[: self.intervals,  np.isin(Nodel_int, self.coverage_int)]
            self.TSPV =   TSPV[: self.intervals,   np.isin(PVl_int,   self.coverage_int)]
            self.TSWind = TSWind[: self.intervals, np.isin(Windl_int, self.coverage_int)]
            
            self.CHydro =    CHydro[   np.isin(Nodel_int, self.coverage_int)]
            self.CBio =      CBio[     np.isin(Nodel_int, self.coverage_int)]
            self.CBaseload = CBaseload[np.isin(Nodel_int, self.coverage_int)]
            self.CPeak =     CPeak[    np.isin(Nodel_int, self.coverage_int)]
        
            if int64(0) not in self.coverage_int:
                self.MLoad[:, np.where(self.coverage_int == 3)[0][0]] /= 0.9
        
            self.Nodel_int = Nodel_int[np.isin(Nodel_int, self.coverage_int)]
            self.PVl_int =   PVl_int[  np.isin(PVl_int,   self.coverage_int)]
            self.Windl_int = Windl_int[np.isin(Windl_int, self.coverage_int)]
        
            with objmode():
                (self.basic_network, 
                 self.network, 
                 self.network_mask, 
                 self.trans_mask, 
                 self.directconns, 
                 self.triangulars,
                ) = generate_network(basic_network, self.Nodel_int)
            
        # firstyear, finalyear, timestep = (2020, 2020 + years - 1, 1)
    
        self.nhvi = self.network_mask.sum()
        self.nodes = len(self.Nodel_int)
        
        self.pzones = len(self.PVl_int)
        self.wzones = len(self.Windl_int)
        self.pidx = self.pzones
        self.widx = self.pidx + self.wzones
        self.spidx = self.widx + self.nodes
        self.seidx = self.spidx + self.nodes
        
        self.energy = self.MLoad.sum() * 1000 * self.resolution / self.years  # MWh p.a.
        
        self.lb = np.array(
            [0.0] * self.pzones + 
            [0.0] * self.wzones + 
            [0.0] * self.nodes + 
            [0.0] * self.nodes + 
            [0.0] * self.nhvi
            )
        self.ub = np.array(
            [24.0]  * self.pzones + 
            [24.0]  * self.wzones + 
            [24.0]  * self.nodes + 
            [600.0] * self.nodes + 
            [20.0]  * self.nhvi
            )
        
        tspvmean = np.array([col.mean() for col in self.TSPV.T])
        tswindmean = np.array([col.mean() for col in self.TSWind.T])
        mloadmax = np.array([array_max(col) for col in self.MLoad.T])
        self.x0 = np.concatenate(
            (
                self.MLoad.sum() / self.intervals * 0.75 / self.pzones / tspvmean,
                self.MLoad.sum() / self.intervals * 0.75 / self.wzones / tswindmean,
                mloadmax * 1,
                mloadmax * 36,
                np.repeat(array_max(mloadmax) * 0.6, self.nhvi),
            )
        )
        self.x0 = np.minimum(self.ub, self.x0)
    
#%%

# Specify the types for jitclass
solution_spec = [
    ("x", float64[:]),
    ("scenario", int64),
    ("intervals", int64),
    ("nodes", int64),
    ("nhvi", int64),
    ("resolution", float64),
    ("years", int64),
    ("efficiency", float64),
    ("energy", float64),
    ("Flex_res", float64),
    ("Nodel_int", int64[:]),
    # ('PVl_int', int64[:]),
    # ('Windl_int', int64[:]),
    ("networksteps", int64),
    ("network_mask", boolean[:]),
    ("network", int64[:, :, :, :]),
    ("basic_network", int64[:, :]),
    ("triangulars", int64[:]),
    # ("directconns", int64[:, :]),
    # Capacities in GW/GWh
    ("CPV", float64[:]),
    ("CWind", float64[:]),
    ("CPHP", float64[:]),
    ("CPHS", float64[:]),
    ("CHVI", float64[:]),
    ("CBaseload", float64[:]),
    ("CPeak", float64[:]),
    ("CHydro", float64[:]),
    ("CBio", float64[:]),
    # Nodally diaggregated operations in GW/GWh
    ("MFlexible", float64[:, :]),
    ("MDischarge", float64[:, :]),
    ("MCharge", float64[:, :]),
    ("MStorage", float64[:, :]),
    ("MDeficit", float64[:, :]),
    ("MSpillage", float64[:, :]),
    ("MNetload", float64[:, :]),
    ("MImport", float64[:, :]),
    ("MPV", float64[:, :]),
    ("MWind", float64[:, :]),
    ("MLoad", float64[:, :]),
    ("MBaseload", float64[:, :]),
    ("MHydro", float64[:, :]),
    ("MBio", float64[:, :]),
    ("MUnbalanced", float64[:,:]),
    # Transmission
    ("TDC", float64[:, :]),
    ("Topology", float64[:, :]),
    ("trans_mask", boolean[:, :]),
    ("TImport", float64[:, :, :]),
    ("TExport", float64[:, :, :]),
    #Objectives
    ("Penalties", float64),
    ("LCOE", float64),
    ("LCOG", float64),
    ("LCOB", float64),
    ("LCOSP", float64),
    ("LCOSB", float64),
    ("LCOBS", float64),
    ("LCOBT", float64),
    ("LCOBL", float64),
    ("CAPEX", float64),
    ("OPEX", float64),
    
    ("cache_primary_donors", types.DictType(int64, int64[:, :])),
    ("cache_secondary_donors", types.DictType(int64, int64[:, :, :])),
    ("cache_tertiary_donors", types.DictType(int64, int64[:, :, :])),
    ("cache_quaternary_donors", types.DictType(int64, int64[:, :, :])),
    
    ("profiling", boolean),
    ('profile_overhead', float64),
    # time profiling
    ("time_transmission", int64),
    ("time_backfill", int64),
    ("time_basic", int64),
    ("time_interconnection0", int64),
    ("time_interconnection1", int64),
    ("time_interconnection2", int64),
    ("time_interconnection3", int64),
    ("time_storage_behavior", int64),
    ("time_storage_behaviort", int64),
    ("time_spilldef", int64),
    ("time_spilldeft", int64),
    ("time_update_soc", int64),
    ("time_update_soct", int64),
    ("time_unbalancedt", int64),
    ("time_unbalanced", int64),
    
    ("calls_transmission", int64),
    ("calls_backfill", int64),
    ("calls_basic", int64),
    ("calls_interconnection0", int64),
    ("calls_interconnection1", int64),
    ("calls_interconnection2", int64),
    ("calls_interconnection3", int64),
    ("calls_storage_behavior", int64),
    ("calls_storage_behaviort", int64),
    ("calls_spilldef", int64),
    ("calls_spilldeft", int64),
    ("calls_update_soc", int64),
    ("calls_update_soct", int64),
    ("calls_unbalancedt", int64),
    ("calls_unbalanced", int64),
]
@jitclass(solution_spec)
class Solution:
    def __init__(
            self, 
            x: np.ndarray, 
            sd: Solution_data
            ):
        assert len(x) == len(sd.lb)

        self.x = x

        self.scenario = sd.scenario
        self.nodes = sd.nodes
        self.resolution = sd.resolution
        self.efficiency = sd.efficiency
        self.years = sd.years
        self.intervals = sd.intervals
        self.energy = sd.energy
        
        self.Nodel_int = sd.Nodel_int
        # self.PVl_int, self.Windl_int = sd.PVl_int, sd.Windl_int
        self.network_mask = sd.network_mask
        self.network = sd.network
        self.basic_network = sd.basic_network
        # self.directconns = sd.directconns
        self.triangulars = sd.triangulars
        self.networksteps = np.where(self.triangulars == self.network.shape[2])[0][0]
        self.trans_mask = sd.trans_mask

        self.nhvi = self.network_mask.sum()

        self.Flex_res = 20000 / self.resolution * self.years


        self.CPV =   x[        : sd.pidx]
        self.CWind = x[sd.pidx : sd.widx]
        self.CPHP =  x[sd.widx : sd.spidx]
        self.CPHS =  x[sd.spidx: sd.seidx]
        self.CHVI =  x[sd.seidx: ]
        self.CBaseload = sd.CBaseload
        self.CPeak =     sd.CPeak
        self.CHydro =    sd.CHydro
        self.CBio =      sd.CBio
        
        self.MLoad = sd.MLoad
        self.MPV = np.zeros((self.intervals, self.nodes))
        self.MWind = np.zeros((self.intervals, self.nodes))
        for i, n in enumerate(self.Nodel_int):
            self.MPV[:, i] += (sd.TSPV[:self.intervals, sd.PVl_int == n] * self.CPV[sd.PVl_int == n]).sum(axis=1)
            self.MWind[:, i] += (sd.TSWind[:self.intervals, sd.Windl_int == n] * self.CWind[sd.Windl_int == n]).sum(axis=1)

        self.cache_primary_donors = TypedDict.empty(int64, int64[:, :])
        self.cache_secondary_donors = TypedDict.empty(int64, int64[:, :, :])
        self.cache_tertiary_donors = TypedDict.empty(int64, int64[:, :, :])
        self.cache_quaternary_donors = TypedDict.empty(int64, int64[:, :, :])
        
        self.profile_overhead=0.0
        self.profiling = sd.profiling
        if self.profiling:
            self.time_transmission = 0
            self.time_backfill = 0
            self.time_basic = 0
            self.time_interconnection0 = 0
            self.time_interconnection1 = 0
            self.time_interconnection2 = 0
            self.time_interconnection3 = 0
            self.time_storage_behavior = 0
            self.time_storage_behaviort = 0
            self.time_spilldef = 0
            self.time_spilldeft = 0
            self.time_update_soc = 0
            self.time_update_soct = 0
            self.time_unbalancedt = 0
            self.time_unbalanced = 0
            
            self.calls_transmission = 0
            self.calls_backfill = 0
            self.calls_basic = 0
            self.calls_interconnection0 = 0
            self.calls_interconnection1 = 0
            self.calls_interconnection2 = 0
            self.calls_interconnection3 = 0
            self.calls_storage_behavior = 0
            self.calls_storage_behaviort = 0
            self.calls_spilldef = 0
            self.calls_spilldeft = 0
            self.calls_update_soc = 0
            self.calls_update_soct = 0
            self.calls_unbalancedt = 0
            self.calls_unbalanced = 0

            self.profile_overhead = 0.0
            for _ in range(10000):
                start=cclock()
                self.calls_transmission += 1
                self.profile_overhead += cclock()-start
            self.calls_transmission += 1
            self.profile_overhead/=10000
        
    def _instantiate_operations(self):
        self.MNetload = self.MLoad - self.MPV - self.MWind - self.CBaseload
        self.MUnbalanced = self.MNetload.copy()
        self.MDeficit, self.MSpillage = np.maximum(0, self.MNetload), -np.minimum(0, self.MNetload)

        self.MFlexible = np.zeros((self.intervals, self.nodes), dtype=np.float64)

        self.MDischarge = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        self.MCharge = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        self.MStorage = np.zeros((self.intervals, self.nodes), dtype=np.float64)
        self.MStorage[-1] = 0.5 * self.CPHS

        self.TImport = np.zeros((self.intervals, self.nodes, self.nhvi), dtype=np.float64)
        self.TExport = np.zeros((self.intervals, self.nodes, self.nhvi), dtype=np.float64)
        self.TDC = np.zeros((self.intervals, self.nodes), dtype=np.float64)


#%% 

@njit
def Evaluate(S, cost_model):
    S._instantiate_operations()
    Simulate(S)

    S.Penalties = np.maximum(0, S.MDeficit.sum())*1000  # MWh/resolution

    CHVI = np.zeros(len(S.network_mask), dtype=np.float64)
    CHVI[S.network_mask] = S.CHVI

    cost = np.array(
        [
            # generation capex
            S.CPV.sum() * cost_model.pv[0],
            S.CWind.sum() * cost_model.onsw[0],
            0,  # S.CGas.sum()  * cost_model.gas[0],
            (S.CHydro.sum() + S.CBio.sum() + S.CBaseload.sum()) * cost_model.hydro[0],
            # generation fom
            S.CPV.sum() * cost_model.pv[1],
            S.CWind.sum() * cost_model.onsw[1],
            0,  # S.CGas.sum()  * cost_model.gas[1],
            (S.CHydro.sum() + S.CBio.sum() + S.CBaseload.sum()) * cost_model.hydro[1],
            # generation vom
            # pv, onsw, battery are 0
            0,  # S.GGas.sum() * S.resolution / S.years * cost_model.gas[2],
            (S.MFlexible.sum() + S.CBaseload.sum() * S.intervals) * S.resolution / S.years * cost_model.hydro[2],
            # storage
            S.CPHP.sum() * cost_model.phes[0],
            S.CPHS.sum() * cost_model.phes[1],
            S.CPHP.sum() * cost_model.phes[2],
            S.MDischarge.sum() * S.resolution / S.years * cost_model.phes[3],
            cost_model.phes[4],
        ]
        +
        # transmission network
        list(
            (
                S.CPV.sum()
                + S.CWind.sum()
                +
                # S.CGas.sum() +
                S.CHydro.sum()
                + S.CBio.sum()
            )
            * cost_model.ac
        )
        + list((CHVI * cost_model.hvi).sum(axis=1))
    )

    # Levelised Costs of:
    # Electricity
    S.LCOE = cost.sum() / S.energy
    # Generation
    S.LCOG = cost[:10].sum() / (
        1000
        * S.resolution
        / S.years
        * (
            S.MPV.sum()
            + S.MWind.sum()
            # +S.MGas.sum()
            + S.MFlexible.sum()
            + S.CBaseload.sum() * S.intervals
        )
    )
    # Storage
    # S.LCOSP = zero_safe_division(cost[10:15].sum(), S.MDischarge.sum()*S.resolution/S.years)
    # Balancing - Storage
    S.LCOBS = cost[10:15].sum() / S.energy
    # Balancing - Transmission
    S.LCOBT = cost[15:].sum() / S.energy
    # Balancing - Spillage
    S.LCOBL = S.LCOE - S.LCOG - S.LCOBS - S.LCOBT
    S.LCOB = S.LCOBS + S.LCOBT + S.LCOBL

    S.CAPEX = sum([cost[i] for i in [0, 1, 2, 3, 10, 11, 15, 18]]) / S.energy
    S.OPEX = S.LCOE - S.CAPEX

    return S.LCOE, S.Penalties

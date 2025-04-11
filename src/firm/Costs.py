import numpy as np
from numba import boolean, float64, int64, njit  # type: ignore
from numba.experimental import jitclass  # type: ignore
from firm.Input import lengths, undersea_mask

USD_to_AUD = 1 / 0.65  # AUD to USD where necessary
discount_rate = 0.0599  # Real discount rate - same as gencost
USD_inflation = 1.18  # 2020->2023
AUD_inflation = 1.16  # 2020->2023
MWh_per_GJ = 0.27778
carbon_price = 0  # AUD/tCO2e
tCO2e_per_GJ_gas = 0.05
tCO2e_per_GJ_coal = 0.1


# costs come from Apx Table B.9 of GenCost 2023-24
# year = 2023
# ==============================================================================
# utility solar
csiro_pv = (
    1526,  # capex AUD/kW
    17,  # fom   AUD/kW p.a.
    0,  # vom   AUD/MWh
    30,  # life  years
)

# onshore wind
csiro_onsw = (
    3038,  # capex AUD/kW
    25,  # fom   AUD/kW p.a.
    0,  # vom   AUD/MWh
    25,  # life  years
)

# offshore wind
csiro_offw = (
    5545,  # capex AUD/kW
    149.9,  # fom   AUD/kW
    0,  # vom   AUD/MWh
    25,  # life  years
)

# large open cycle gas
csiro_gas = (
    943,  # capex AUD/kW
    10.2,  # fom   AUD/kW p.a.
    7.3,  # vom   AUD/MWh
    16.5 / 0.33 / MWh_per_GJ,  # fuel AUD/MWh
    tCO2e_per_GJ_gas / MWh_per_GJ * carbon_price,  # carbon intensity tCO2e/MWh
    25,  # life  years
)

# black coal
csiro_coal = (
    5616,  # capex AUD/kW
    53.2,  # fom   AUD/kW p.a.
    4.2,  # vom   AUD/MWh
    7.8 / 0.42 / MWh_per_GJ,  # fuel AUD/MWh
    tCO2e_per_GJ_coal / MWh_per_GJ * carbon_price,  # carbon intensity tCO2e/MWh
    30,  # life  years
)

# costs adjusted for inflation but otherwise unchanged from Lu et al. 2021 https://doi.org/10.1016/j.energy.2020.119678
# ==============================================================================
# TODO: find costs for this
interconnector = (
    160 * AUD_inflation,  # capex AUD/kW
    1.6 * AUD_inflation,  # fom   AUD/kW p.a.
    0,  # vom   AUD/MWh
    30,  # life  years
)

# undersea costs includer converter
hv_undersea = (
    4000 * AUD_inflation,  # capex AUD/MW-km
    40 * AUD_inflation,  # fom   AUD/MW-km p.a.
    0,  # vom   AUD/MWh
    30,  # life  years
)

hvac = (
    1500 * AUD_inflation,  # capex AUD/MW-km
    15 * AUD_inflation,  # fom   AUD/MW-km p.a.
    0,  # vom   AUD/MWh
    50,  # life  years
)

# costs from re100 cost model - Class AA site
# 1 GW / 160 GWh
# 700 m head, 10 km separation
# 8.0 W/R
# ==============================================================================
phes = (
    1164 * USD_inflation * USD_to_AUD,  # capex AUD/kW
    15 * USD_inflation * USD_to_AUD,  # capex AUD/kWh
    8.21 * USD_inflation * USD_to_AUD,  # fom AUD/kW p.a.
    0.6 * USD_inflation * USD_to_AUD,  # vom AUD/MWh
    112000 * USD_inflation * USD_to_AUD,  # AUD per replace
    50,  # replace lifetime
    100,  # life years
)

# same O&M as PHES, but no capital
hydro = (
    0,  # capex (existing only)
    8.21 * USD_inflation * USD_to_AUD,  # fom AUD/kW p.a. #same as phes (approx)
    0.3 * USD_inflation * USD_to_AUD / 2,  # vom AUD/MWh #half phes (one-way trip) (approx)
    50,  # life years
)

# Battery costs
# Lazard LCOE+ 2 Hour utility battery
battery = (
    45 * USD_to_AUD,  # capex AUD/kW
    294 * USD_to_AUD,  # capex AUD/kWh
    5.35 * USD_to_AUD,  # fom AUD/kWh
    0,  # vom
    10,  # life years
)


@njit
def present_value(life, dr):
    return (1 - (1 + dr) ** (-1 * life)) / dr


@njit
def annualization(capex, fom, vom, life, dr):
    """
    Calculate annualized costs parametrically for power and energy
    Input:
        capex - $/kW
        fom   - $/kW p.a.
        vom   - $/MWh
        life  - years
        dr    - %
    Output:
        discounted capex cost factor ($ p.a. / GW)
        fom cost factor ($ p.a. / GW)
        vom cost factor ($ p.a. / MWh p.a.)
    """
    return np.array(
        [
            1_000_000 * capex / present_value(life, dr),  # $ p.a./GW
            1_000_000 * fom,  # $ p.a./GW
            1000 * vom,  # $ p.a./GWh p.a.
        ],
        np.float64,
    )


@njit
def annualization_transmission(capex, fom, vom, life, d, dr):
    """
    Calculate annualized costs parametrically for power and energy, for transmission lines only
    Input:
        capex - $/MW-km
        fom   - $/MW-km p.a.
        vom   - $/MWh
        life  - years
        d     - km
        dr    - %
    Output:
        discounted capex cost factor ($ p.a. / GW)
        fom cost factor ($ p.a. / GW)
        vom cost factor ($ p.a. / MWh p.a.)
    """
    return np.array(
        [
            d * capex * 1000 / present_value(life, dr),  # $ p.a./GW
            d * fom * 1000,  # $ p.a./GW
            vom * 1000,  # $ p.a./GWh p.a.
        ]
    )


@njit
def annualization_phes(capex_p, capex_e, fom, vom, replace_cost, replace_life, life, dr):
    """Calculate annualized costs parametrically for power and energy, for PHES only
    capex_p, fom: AUD/kW
    capex_e: AUD/kWh
    fom: AUD/kW p.a.
    vom: AUD/MWh p.a.
    replace: AUD per replace
    replace_life: years"""
    pv = present_value(life, dr)
    return np.array(
        [
            capex_p * 1_000_000 / pv,  # capex $ p.a./GW
            capex_e * 1_000_000 / pv,  # capex $ p.a./GWh
            fom * 1_000_000,  # fom $ p.a./GW
            vom * 1000,  # vom $ p.a./GWh p.a.
            # TODO: check this
            replace_cost
            * ((1 + dr) ** (-1 * replace_cost) + (1 + dr) ** (-1 * replace_life * 2))
            / pv,  # replace capex $p.a.
        ]
    )


@njit
def annualization_battery(capex_p, capex_e, fom, vom, life, dr):
    """Calculate annualized costs parametrically for power and energy, for batteries only
    capex_p, fom: AUD/kW
    capex_e: AUD/kWh
    fom: AUD/kWh p.a.
    vom: AUD/MWh
    life: years"""
    pv = present_value(life, dr)
    return np.array(
        [
            capex_p * 1_000_000 / pv,  # capex $ p.a./GW
            capex_e * 1_000_000 / pv,  # capex $ p.a./GWh
            fom * 1_000_000,  # fom $ p.a./GW
            vom * 1000,  # vom $ p.a./GWh p.a.
        ]
    )


@njit
def annualization_fossils(capex, fom, vom, fuel, carbon, life, dr):
    """
    Calculate annualized costs parametrically for power and energy
    Input:
        capex - $/kW
        fom   - $/kW p.a.
        vom   - $/MWh
        fuel  - $/MWh
        life  - years
        dr    - %
    Output:
        discounted capex cost factor ($ p.a. / GW)
        fom cost factor ($ p.a. / GW)
        vom cost factor ($ p.a. / MWh p.a.)
    """
    return np.array(
        [
            1_000_000 * capex / present_value(life, dr),  # $ p.a./GW
            1_000_000 * fom,  # $ p.a./GW
            1000 * vom + 1000 * fuel + 1000 * carbon,  # $ p.a./GWh p.a.
        ],
        np.float64,
    )


@jitclass(
    [
        ("carbon_price", float64),
        ("dr", float64),
        ("pv", float64[:]),
        ("onsw", float64[:]),
        ("offw", float64[:]),
        ("gas", float64[:]),
        ("coal", float64[:]),
        ("hydro", float64[:]),
        ("phes", float64[:]),
        ("battery", float64[:]),
        ("hvac", float64[:]),
        ("hvi", float64[:]),
        ("hvu", float64[:]),
        ("scenario", int64),
        ("lengths", int64[:]),
        ("undersea_mask", boolean[:]),
        ("network_mask", boolean[:]),
    ]
)
class Raw_Costs:
    def __init__(
        self,
        solution_data
    ):
        self.scenario = solution_data.scenario
        self.network_mask = solution_data.network_mask

        self.pv = np.array(csiro_pv, np.float64)
        self.onsw = np.array(csiro_onsw, np.float64)
        self.offw = np.array(csiro_offw, np.float64)
        self.gas = np.array(csiro_gas, np.float64)
        self.coal = np.array(csiro_coal, np.float64)
        self.hydro = np.array(hydro, np.float64)
        self.phes = np.array(phes, np.float64)
        self.battery = np.array(battery, np.float64)
        self.hvac = np.array(hvac, np.float64)
        self.hvi = np.array(interconnector, np.float64)
        self.hvu = np.array(hv_undersea, np.float64)

        self.dr = discount_rate
        self.UpdateCarbonPrice(carbon_price)

    def UpdateCarbonPrice(self, price):
        self.gas[4] = tCO2e_per_GJ_gas / MWh_per_GJ * price
        self.coal[4] = tCO2e_per_GJ_coal / MWh_per_GJ * price

    def CostFactors(self):
        return Cost_Factors(self)


@jitclass(
    [
        ("pv", float64[:]),
        ("onsw", float64[:]),
        ("offw", float64[:]),
        ("gas", float64[:]),
        ("coal", float64[:]),
        ("hydro", float64[:]),
        ("phes", float64[:]),
        ("battery", float64[:]),
        ("ac", float64[:]),
        ("hvi", float64[:, :]),
    ]
)
class Cost_Factors:
    def __init__(self, raw_costs):
        self.pv = annualization(raw_costs.pv[0], raw_costs.pv[1], raw_costs.pv[2], raw_costs.pv[3], raw_costs.dr)
        self.onsw = annualization(
            raw_costs.onsw[0], raw_costs.onsw[1], raw_costs.onsw[2], raw_costs.onsw[3], raw_costs.dr
        )
        self.offw = annualization(
            raw_costs.offw[0], raw_costs.offw[1], raw_costs.offw[2], raw_costs.offw[3], raw_costs.dr
        )

        self.gas = annualization_fossils(
            raw_costs.gas[0],
            raw_costs.gas[1],
            raw_costs.gas[2],
            raw_costs.gas[3],
            raw_costs.gas[4],
            raw_costs.gas[5],
            raw_costs.dr,
        )
        self.coal = annualization_fossils(
            raw_costs.coal[0],
            raw_costs.coal[1],
            raw_costs.coal[2],
            raw_costs.coal[3],
            raw_costs.coal[4],
            raw_costs.coal[5],
            raw_costs.dr,
        )

        self.hydro = annualization(
            raw_costs.hydro[0], raw_costs.hydro[1], raw_costs.hydro[2], raw_costs.hydro[3], raw_costs.dr
        )
        self.phes = annualization_phes(
            raw_costs.phes[0],
            raw_costs.phes[1],
            raw_costs.phes[2],
            raw_costs.phes[3],
            raw_costs.phes[4],
            raw_costs.phes[5],
            raw_costs.phes[6],
            raw_costs.dr,
        )

        self.battery = annualization_battery(
            raw_costs.battery[0],
            raw_costs.battery[1],
            raw_costs.battery[2],
            raw_costs.battery[3],
            raw_costs.battery[4],
            raw_costs.dr,
        )

        self.ac = annualization_transmission(
            raw_costs.hvac[0], raw_costs.hvac[1], raw_costs.hvac[2], raw_costs.hvac[3], 20, raw_costs.dr
        )

        self.hvi = np.zeros((len(raw_costs.network_mask), 3), np.float64)
        if raw_costs.scenario >= 21:
            for i, undersea in enumerate(undersea_mask):
                if raw_costs.network_mask[i] is False:
                    continue
                if undersea:
                    self.hvi[i] = annualization_transmission(
                        raw_costs.hvu[0],
                        raw_costs.hvu[1],
                        raw_costs.hvu[2],
                        raw_costs.hvu[3],
                        lengths[i],
                        raw_costs.dr,
                    )  # vom is 0
                else:
                    self.hvi[i] = annualization(
                        raw_costs.hvi[0], raw_costs.hvi[1], raw_costs.hvi[2], raw_costs.hvi[3], raw_costs.dr
                    )
        self.hvi = self.hvi.T

if __name__=='__main__':
    from firm.Parameters import Parameters
    from firm.Input import Solution_data
    parameters = Parameters(21, 1, False)
    sd = Solution_data(*parameters)
    costs = Raw_Costs(sd).CostFactors()
    
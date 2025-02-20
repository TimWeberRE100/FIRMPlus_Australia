import numpy as np

# AUD to USD conversion 1 : 0.7 where necessary
USD_to_AUD = 1.43 # AUD to USD where necessary
discount_rate = 0.0599 # Real discount rate - same as gencost
USD_inflation = 1.18 # 2020->2023
AUD_inflation = 1.16 # 2020->2023
GJ_to_MWh = 0.27778

## costs come from Apx Table B.9 of GenCost 2023-24 
## year = 2023
#==============================================================================
# utility solar
csiro_pv = (
    1526,   # capex AUD/kW 
    17,     # fom   AUD/kW p.a.
    0,      # vom   AUD/MWh p.a.
    30,     # life  years
    )
irena_pv = (
    989 * USD_to_AUD,  # capex AUD/kW - AUS average
    18.2 * USD_to_AUD, # fom   AUD/kW - IRENA assumption for OECD, 
    0, 
    30,
    )

# onshore wind
csiro_onsw = (
    3038,   # capex AUD/kW 
    25,     # fom   AUD/kW p.a.
    0,      # vom   AUD/MWh p.a.
    25,     # life  years
    )
irena_onsw = (
    1572 * USD_to_AUD, # capex AUD/kW - AUS average
    42.9 * USD_to_AUD, # fom   AUD/kW p.a. - average of major markets' weighted averages
    0, 
    25
    )

# offshore wind
csiro_offw = (
    5545,   # capex AUD/kW 
    149.9,  # fom   AUD/kW p.a.
    0,      # vom   AUD/MWh p.a.
    25,     # life  years
    )
irena_offw = (
    2800 * USD_to_AUD, # capex AUD/kW - global weighted average
    92.5 * USD_to_AUD, # fom   AUD/kW p.a. - middle of 2023 range from (Wood Mackenzie, 2023d) pp 123
    0, 
    25,
    )
# Gas 
# costs from GenCost 2024 - gas open cycle (large) (peaking)
gas = (
    943,    # capex AUD/kW
    10.2,   # fom   AUD/kW p.a.
    7.3+16.5/0.33*GJ_to_MWh, # vom + fuel  AUD/MWh p.a.
    25,     # life  years
    )

battery_capexes = (
    922, # 1 hour battery - # AUD/kWh
    659, # 2 hour
    528, # 4 hour
    460, # 8 hour
    427, # 12 hour
    )

battery = (
    0,  # fom  AUD/kW p.a.
    0,  # vom  AUD/MWh
    20, # life years
    )

## costs unchanged from Lu et al. 2021 https://doi.org/10.1016/j.energy.2020.119678
#==============================================================================
hvdc_overhead = (
    320 * AUD_inflation,  # capex AUD/MW-km
    3.2 * AUD_inflation,  # fom   AUD/MW-km p.a.
    0,                    # vom   AUD/MWh-km p.a.
    50,                   # life  years
    )

converter = (
    160 * AUD_inflation,  # capex AUD/kw
    1.6 * AUD_inflation,  # fom   AUD/kw p.a.
    0,                    # vom   AUD/MWh p.a.
    30,                   # life  years
    )

# undersea costs includer converter
hvdc_undersea = (
    4000 * AUD_inflation, # capex AUD/kw
    40   * AUD_inflation, # fom   AUD/kw p.a.
    0,                    # vom
    30,                   # life  years
    )

hvac = (
    1500 * AUD_inflation, # capex AUD/MW-km
    15   * AUD_inflation, # fom   AUD/MW-km p.a.
    0,                    # vom 
    50,                   # life  years
    )

hydro_purchase = 50 # AUD/MWh p.a.

## costs from re100 cost model - Class A site
#==============================================================================
phes = (
    530    * USD_inflation * USD_to_AUD, # capex AUD/kW
    47     * USD_inflation * USD_to_AUD, # capex AUD/kWh
    8.21   * USD_inflation * USD_to_AUD, # fom AUD/kW p.a.
    0.3    * USD_inflation * USD_to_AUD, # vom AUD/MWh p.a.
    112000 * USD_inflation * USD_to_AUD, # AUD per replace
    50,  # replace lifetime
    100, # life years
    )
## functions
#==============================================================================
def present_value(dr, life):
    """ value of future annual costs in current currency """
    return (1-(1+dr)**(-life))/dr

def annualization(capex, fom, vom, life, dr, p, e):
    """ Calculate annualized costs for capacity p and annual generation e.
        capex: $/kW
        fom: $/kW p.a.
        vom: $/MWh p.a.
        p: GW
        e: MWh """
    pv = present_value(dr, life)
    return p * pow(10,6) * capex / pv + p * pow(10,6) * fom + e * vom

def annualization_transmission(capex, transformer_capex, fom, vom, life, dr, p, e, d):
    """ Calculate annualized costs for capacity p and annual generation e, for transmission lines only.
        capex: $/MW-km
        transformer_capex: $/MW
        fom: $/MW-km p.a.
        vom: $/MWh p.a.
        p: GW
        e: MWh
        d: km """
    pv = present_value(dr, life)
    return (p * pow(10,3) * d * capex + p * pow(10,3) * transformer_capex) / pv + p * pow(10,3) * d * fom + e * vom

def annualization_constants(capex, fom, vom, life, dr):
    """ Calculate annualized costs parametrically for power and energy """
    pv = present_value(dr, life)
    return pow(10,6) * capex / pv + pow(10,6) * fom, vom

def annualization_transmission_constants(capex, fom, vom, life, d, dr):
    """ Calculate annualized costs parametrically for power and energy, for transmission lines only"""
    pv = present_value(dr, life)
    return d * capex * pow(10,3) / pv + d * fom * pow(10,3), vom

def annualization_phes_constants(capex_p, capex_e, fom, vom, replace_cost, replace_life, life, dr):
    """ Calculate annualized costs parametrically for power and energy, for PHES only 
    capex_p, fom: USD/kW
    capex_e: USD/kWh
    vom: USD/MWh
    replace: USD per replace
    replace_life: years """
    pv = present_value(dr, life)
    return (capex_p* pow(10,6) / pv + fom * pow(10,6), # * GW = cost
            capex_e * pow(10,6) / pv, # * GWh = cost
            vom * pow(10,3),# * (MWh discharge p.a.) = cost
            replace_cost * ((1+dr)**(-1*replace_cost) + (1+dr)**(-1*replace_life*2)) / pv # *1 = cost
            ) 

## processed cost factors
#==============================================================================
class cost_factors:
    def __init__(self, source, DClengths, undersea_mask):
        if source == 'csiro':
            self.pv    = annualization_constants(*csiro_pv,   discount_rate)[0] #vom is 0
            self.onsw  = annualization_constants(*csiro_onsw, discount_rate)[0] #vom is 0
            self.offw  = annualization_constants(*csiro_offw, discount_rate)[0] #vom is 0
        if source == 'irena':
            self.pv    = annualization_constants(*irena_pv,   discount_rate)[0] #vom is 0
            self.onsw  = annualization_constants(*irena_onsw, discount_rate)[0] #vom is 0
            self.offw  = annualization_constants(*irena_offw, discount_rate)[0] #vom is 0
        self.gas = annualization_constants(*gas, discount_rate)
        
        self.ac    = annualization_transmission_constants(*hvac, 20, discount_rate)[0] #vom is 0
        
        self.phes  = annualization_phes_constants(*phes, discount_rate)
        self.batte = np.array([annualization_constants(capex, *battery, discount_rate)[0] for capex in battery_capexes]) # vom is 0 
        
        self.hvdc = np.zeros(len(DClengths), float)
        for i, undersea in enumerate(undersea_mask):
            if undersea:
                self.hvdc[i] = annualization_transmission_constants(*hvdc_undersea, DClengths[i], discount_rate)[0] # vom is 0
            else: 
                self.hvdc[i] = annualization_transmission_constants(*hvdc_overhead, DClengths[i], discount_rate)[0] # vom is 0
                self.hvdc[i] += 2*annualization_constants(*converter, discount_rate)[0]

        self.hydro=hydro_purchase

if __name__ == '__main__':
    from Input import cost_source, DClengths, undersea_mask
    
    costs = cost_factors(cost_source, DClengths, undersea_mask)

import numpy as np
from Input import DClengths, network_mask

# AUD to USD conversion 1 : 0.7 where necessary
curr_conv = 0.7

## Costs, except for transmission and pmped hydro come from Apx Table B.9 iof Gencost 2023-24 (year=2050, assumption=low)

##Transmission costs should be updated for Australia

pv_capex = 583 # AUD/kW GenCost 2023-4
pv_fom = 17 # AUD/kW
pv_vom = 0 # AUD/MWh
pv_lifetime = 30

wind_capex = 1763 # AUD/kW 
wind_fom = 25 # AUD/kW 
wind_vom = 0 # AUD/MWh
wind_lifetime = 25

wind_offshore_capex = 2691 # AUD/kW 
wind_offshore_fom = 149.9 # AUD/kW 
wind_offshore_vom = 0 # AUD/MWh
wind_offshore_lifetime = 25

# HVAC costs from https://www.adb.org/sites/default/files/project-documents/47129/47129-001-tacr-en.pdf
# Based on "Circuit" line, transformer and substation cost and lifetime in Appendix 4
# 500kV line assumed to have 1500 MW capacity, as per Table 9: https://www.adb.org/sites/default/files/project-documents/47129/47129-001-tacr-en.pdf
transmission_hvac_capex = 463000/1500 / curr_conv # (AUD/km) / MW
transmission_hvac_fom = 3.2 / curr_conv # AUD/MW-km p.a.
transmission_hvac_vom = 0 # AUD/MWh p.a.
transmission_hvac_lifetime = 60
transmission_hvac_transformers = 11000 / curr_conv # AUD/MW, transformer cost (AUD/MW)

# HVDC point-to-point costs from https://www.adb.org/sites/default/files/project-documents/47129/47129-001-tacr-en.pdf
# 500kV line assumed to have 3000 MW capacity, as per Table 9: https://www.adb.org/sites/default/files/project-documents/47129/47129-001-tacr-en.pdf
transmission_hvdc_capex = 394000/3000 / curr_conv # (AUD/km) / MW
transmission_hvdc_fom = 3.2 / curr_conv  # AUD/MW-km p.a.
transmission_hvdc_vom = 0 # AUD/MWh p.a.
transmission_hvdc_lifetime = 60

# HVDC converter station
converter_capex = 160  # AUD/kW each
converter_fom = 1.6  # AUD/kW each p.a.
converter_vom = 0 # AUD/MWh p.a.
converter_lifetime = 60

# Assume class A site, scaled to US 2024 dollars
storage_capexP = 530/0.83 / curr_conv # AUD/kW 
stoarge_capexE = 47/0.83 / curr_conv # AUD/kWh
storage_fom = 8.21 / curr_conv # AUD/kW p.a.
storage_vom = 0.3 / curr_conv # AUD/MWh p.a.
storage_replace = 112000 / curr_conv # AUD per replace
replace = 50 # every 50 years
storage_lifetime = 100 #operational life

# battery_capexP = 45 # USD/kW, median Initial Capital Cost AC for 100MW/400MWh battery in Lazard LCOE+: https://www.lazard.com/media/xemfey0k/lazards-lcoeplus-june-2024-_vf.pdf 
# battery_capexE = 221 + 70 # USD/kWh, median Initial Capital Cost DC + EPC costs for 100MW/400MWh battery in Lazard LCOE+: https://www.lazard.com/media/xemfey0k/lazards-lcoeplus-june-2024-_vf.pdf
# battery_fom = 5.25 # USD/kWh, median O&M for 100MW/400MWh battery in Lazard LCOE+: https://www.lazard.com/media/xemfey0k/lazards-lcoeplus-june-2024-_vf.pdf
# battery_vom = 0 # AUD/MWh p.a.
# battery_lifetime = 20

hydro_cost = 50 # AUD/MWh

DR = 0.0599 # Real discount rate - same as gencost

def annualization(capex, fom, vom, life, dr, p, e):
    """ Calculate annualized costs for capacity p and annual generation e.
        capex: $/kW
        fom: $/kW p.a.
        vom: $/MWh p.a.
        p: GW
        e: MWh """
    pv = (1-(1+dr)**(-1*life))/dr
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
    pv = (1-(1+dr)**(-1*life))/dr
    return (p * pow(10,3) * d * capex + p * pow(10,3) * transformer_capex) / pv + p * pow(10,3) * d * fom + e * vom

def annualization_constants(capex, fom, vom, life, dr):
    """ Calculate annualized costs parametrically for power and energy """
    pv = (1-(1+dr)**(-1*life))/dr
    return pow(10,6) * capex / pv + pow(10,6) * fom, vom

def annualization_transmission_constants(capex, transformer_capex, fom, vom, life, dr, d):
    """ Calculate annualized costs parametrically for power and energy, for transmission lines only"""
    pv = (1-(1+dr)**(-1*life))/dr
    return (d * capex * pow(10,3) + transformer_capex * pow(10,3)) / pv + d * fom * pow(10,3), vom

def annualization_phes_constants(capex_p, capex_e, fom, vom, replace_cost, replace_life, life, dr):
    """ Calculate annualized costs parametrically for power and energy, for PHES only 
    capex_p, fom: USD/kW
    capex_e: USD/kWh
    vom: USD/MWh
    replace: USD per replace
    replace_life: years """
        
    pv = (1-(1+dr)**(-1*life))/dr
    
    return (capex_p* pow(10,6) / pv + fom * pow(10,6), # * GW = cost
            capex_e * pow(10,6) / pv, # * GWh = cost
            vom * pow(10,3),# * (MWh discharge p.a.) = cost
            replace_cost * ((1+dr)**(-1*replace_cost) + (1+dr)**(-1*replace_life*2)) / pv # *1 = cost
            ) 

def annualization_battery_constants(capex_p, capex_e, fom, vom, life, dr):
    """ Calculate annualized costs parametrically for power and energy, for batteries only 
    capex_p, fom: USD/kW
    capex_e: USD/kWh
    vom: USD/MWh"""

    pv = (1-(1+dr)**(-1*life))/dr
    return (capex_p * pow(10,6) / pv, 
            capex_e * pow(10,6) / pv + vom * pow(10,6)
            )

pv_costs = annualization_constants(pv_capex, pv_fom, pv_vom, pv_lifetime, DR)[0] #vom is 0
wind_costs = annualization_constants(wind_capex, wind_fom, wind_vom, wind_lifetime, DR)[0] #vom iis 0

ACgen_costs = annualization_transmission_constants(transmission_hvac_capex, transmission_hvac_transformers, 
                                                 transmission_hvac_fom, transmission_hvac_vom, transmission_hvac_lifetime, DR, 20)[0] #vom is 0
storage_costs = annualization_phes_constants(storage_capexP, stoarge_capexE, storage_fom, storage_vom, storage_replace, replace, storage_lifetime, DR)

transmission_costs = np.array([annualization_transmission_constants(transmission_hvdc_capex, 0, transmission_hvdc_fom, transmission_hvdc_vom, transmission_hvdc_lifetime, DR, d)[0]
                               for d in DClengths[network_mask]])
substation_costs = tuple((2*i for i in annualization_constants(converter_capex, converter_fom, converter_vom, converter_lifetime, DR)))[0] # vom is 0



import numpy as np
from Input import DClengths, network_mask

# =============================================================================
# Would be good to revisit these before using any results
#   Taken from FIRM Mekong
# =============================================================================

# AUD to USD conversion 1 : 0.7 where necessary
curr_conv = 0.7

pv_capex = 671 # USD/kW, Mean global cost, IRENA Renewable Power Generation Costs in 2023: https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2023/Aug/IRENA_Renewable_power_generation_costs_in_2022.pdf
pv_fom = 3.6 # USD/kW p.a. Median Asia cost, IRENA Renewable Power Generation Costs in 2022: https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2023/Aug/IRENA_Renewable_power_generation_costs_in_2022.pdf
pv_vom = 0 # USD/MWh p.a.
pv_lifetime = 30

wind_capex = 986 # USD/kW, Mean global onshore wind cost, IRENA Renewable Power Generation Costs in 2023: https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2023/Aug/IRENA_Renewable_power_generation_costs_in_2022.pdf
wind_fom = 29.5 # USD/kW p.a., No great data recent for ASEAN, so just used same assumption as 7th ASEAN Energy Outlook:https://asean.org/wp-content/uploads/2023/04/The-7th-ASEAN-Energy-Outlook-2022.pdf
wind_vom = 0 # USD/MWh p.a.
wind_lifetime = 25

wind_offshore_capex = 2370 # USD/kW, Mean global offshore wind cost, IRENA Renewable Power Generation Costs in 2022: https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2023/Aug/IRENA_Renewable_power_generation_costs_in_2022.pdf
wind_offshore_fom = 67 # USD/kW p.a., No great data recent for ASEAN, so just used same assumption as 7th ASEAN Energy Outlook:https://asean.org/wp-content/uploads/2023/04/The-7th-ASEAN-Energy-Outlook-2022.pdf
wind_offshore_vom = 0 # USD/MWh p.a.
wind_offshore_lifetime = 25

# HVAC costs from https://www.adb.org/sites/default/files/project-documents/47129/47129-001-tacr-en.pdf
# Based on "Circuit" line, transformer and substation cost and lifetime in Appendix 4
# 500kV line assumed to have 1500 MW capacity, as per Table 9: https://www.adb.org/sites/default/files/project-documents/47129/47129-001-tacr-en.pdf
transmission_hvac_capex = 463000/1500 # (USD/km) / MW
transmission_hvac_fom = 3.2 * curr_conv # USD/MW-km p.a.
transmission_hvac_vom = 0 # USD/MWh p.a.
transmission_hvac_lifetime = 60
transmission_hvac_transformers = 11000 # USD/MW, transformer cost (USD/MW)

# HVDC point-to-point costs from https://www.adb.org/sites/default/files/project-documents/47129/47129-001-tacr-en.pdf
# 500kV line assumed to have 3000 MW capacity, as per Table 9: https://www.adb.org/sites/default/files/project-documents/47129/47129-001-tacr-en.pdf
transmission_hvdc_capex = 394000/3000 # (USD/km) / MW
transmission_hvdc_fom = 3.2 * curr_conv # USD/MW-km p.a.
transmission_hvdc_vom = 0 # USD/MWh p.a.
transmission_hvdc_lifetime = 60

# HVDC converter station
converter_capex = 160 * curr_conv # USD/kW each
converter_fom = 1.6 * curr_conv # USD/kW each p.a.
converter_vom = 0 # USD/MWh p.a.
converter_lifetime = 60

# Gas 
# costs from GenCost 2024 - open cycle (large) 2050 costs
gas_capex = 826 * curr_conv # USD/kW
gas_fom = 10.2 * curr_conv # USD/kW p.a.
gas_vom = 7.3 * curr_conv # $/MWh 
gas_fuel = 13.5 / 0.33 * 0.278 * curr_conv # S/MWh 
gas_lifetime = 25

# Assume class A site, scaled to US 2024 dollars
storage_capexP = 530/0.83 # USD/kW # 
stoarge_capexE = 47/0.83 # USD/kWh
storage_fom = 8.21 # USD/kW p.a.
storage_vom = 0.3 # USD/MWh p.a.
storage_replace = 112000 # USD per replace
replace = 50 # every 50 years
storage_lifetime = 100

# battery_capexP = 45 # USD/kW, median Initial Capital Cost AC for 100MW/400MWh battery in Lazard LCOE+: https://www.lazard.com/media/xemfey0k/lazards-lcoeplus-june-2024-_vf.pdf 
# battery_capexE = 221 + 70 # USD/kWh, median Initial Capital Cost DC + EPC costs for 100MW/400MWh battery in Lazard LCOE+: https://www.lazard.com/media/xemfey0k/lazards-lcoeplus-june-2024-_vf.pdf
# battery_fom = 5.25 # USD/kWh, median O&M for 100MW/400MWh battery in Lazard LCOE+: https://www.lazard.com/media/xemfey0k/lazards-lcoeplus-june-2024-_vf.pdf
# battery_vom = 0 # AUD/MWh p.a.
# battery_lifetime = 20

hydro_cost = 50 # USD/MWh

DR = 0.05 # Real discount rate 

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
gas_costs = annualization_constants(gas_capex, gas_fom, gas_vom + gas_fuel, gas_lifetime, DR)

AC_gen = annualization_transmission_constants(transmission_hvac_capex, transmission_hvac_transformers, 
                                                 transmission_hvac_fom, transmission_hvac_vom, transmission_hvac_lifetime, DR, 20)[0] #vom is 0


storage_costs = annualization_phes_constants(storage_capexP, stoarge_capexE, storage_fom, storage_vom, storage_replace, replace, storage_lifetime, DR)

transmission_costs = np.array([annualization_transmission_constants(transmission_hvdc_capex, 0, transmission_hvdc_fom, transmission_hvdc_vom, transmission_hvdc_lifetime, DR, d)[0]
                               for d in DClengths[network_mask]])
substation_costs = tuple((2*i for i in annualization_constants(converter_capex, converter_fom, converter_vom, converter_lifetime, DR)))[0] # vom is 0



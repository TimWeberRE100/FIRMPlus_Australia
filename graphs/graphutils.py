# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:05:45 2024

@author: u6942852
"""
from ast import literal_eval
import numpy as np
import re
import os



def standardiseNas(obj, na=np.nan, extras=[]):
    return na if str(obj).strip().lower() in ['nan', 'none', '', ' ', '-', '<na>'] + extras else obj

def reSearchWrapper(regex, srchstr, fail_val=None):
    try: 
        return re.search(regex, srchstr).group()
    except AttributeError: 
        return fail_val

def adjust_legend(axs, x, y, xscale=0.95, yscale=1.0, loc='center right'):
    """
    ax should be a matplotlib.pylot axis. Where multiple axes are on the same 
    subplot (i.e. twin axis, not individual subplots) pass the multiple axes as 
    a list.
    """
    if isinstance(axs, list):
        lns, labs = zip(*((items for items in x.get_legend_handles_labels()) for x in axs))
        lns = [l for ln in lns for l in ln]
        labs = [l for lab in labs for l in lab]
        ax = axs[0]
    else: 
        lns, labs = axs.get_legend_handles_labels()
        ax = axs
    
    pairs = dict(zip(labs, lns))
    
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width*xscale, pos.height*yscale])
    
    ax.legend(pairs.values(), pairs.keys(), loc=loc, bbox_to_anchor=(x, y)) 

    return axs

def directory_up():
    os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))

def readPrintedArray(txt):      
    if txt == 'None': return None
    txt = re.sub(r"(?<!\[)\s+(?!\])", r",", txt)
    return np.array(literal_eval(txt), dtype=np.int64)

def manage_nodes(scenario):
    Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
    PVl   = np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
    Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)

    if scenario<=17:
        node = Nodel[scenario % 10]
        pzones = len(np.where(PVl==node)[0])
        wzones = len(np.where(Windl==node)[0])
        coverage = np.array([node])
        
    if scenario>=21:
        coverage = [np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                    np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                    np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                    np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                    np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                    np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                    np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                    np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])][scenario % 10 - 1]
        pzones = len(np.where(np.in1d(PVl, coverage)==True)[0])
        wzones = len(np.where(np.in1d(Windl, coverage)==True)[0])
    
    pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + len(coverage))
    
    return (Nodel, PVl, Windl,), (pzones, wzones,), (pidx, widx, sidx,), (coverage,)
    
def zoneTypeIndx(scenario, wdir=None):
    if wdir is not None: 
        active_dir = os.getcwd()
        os.chdir(wdir)

    pvnames = np.genfromtxt(r'Data\pv.csv', delimiter=',', dtype=str, max_rows=1)[4:]
    windnames = np.genfromtxt(r'Data\wind.csv', delimiter=',', dtype=str, max_rows=1)[4:]
    names = np.append(pvnames, windnames)
    
    node_data = manage_nodes(scenario)
    Nodel, PVl, Windl = node_data[0]
    pidx, widx, sidx = node_data[2]
    coverage = node_data[3]

    if scenario<=17:
        names = names[np.where(np.append(PVl, Windl)==coverage[0])[0]] 
        
    if scenario>=21:
        names = names[np.where(np.in1d(np.append(PVl, Windl), coverage)==True)[0]]

    headers = (['pv-'   + name + ' (GW)' for name in names[:pidx]] +
               ['wind-' + name + ' (GW)' for name in names[pidx:widx]] + 
               ['storage-' + name + ' (GW)' for name in coverage[0]] + 
               ['storage (GWh)'])
    
    if wdir is not None: 
        os.chdir(active_dir)
    return pidx, widx, sidx, headers

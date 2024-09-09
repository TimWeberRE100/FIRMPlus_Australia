# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:39:13 2024

@author: u6942852
"""

import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import warnings

import graphutils as gu

scenario=21
dpi=250


os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))

pidx, widx, sidx, headers = gu.zoneTypeIndx(scenario)

costConstraint=1.02

file =fr"Results\History{scenario}-children.csv"
data = pd.read_csv(file, header=None)


os.chdir('graphs')

#%%
varCols=[f'var{n}' for n in range(1, sidx+2)]
data.columns = ['objective', 'generation', 'cuts', 'LCOE', 'LCOG', 'LCOBS', 'LCOBT', 'LCOBL']+varCols

data['penalties'] = (data['objective']-data['LCOE']).round(5)
data = data[data['penalties'] <0.1]
mincost = data['LCOE'].min()

# fulldata = data.copy()
resolved = data['cuts'].max()
data = data.loc[data['cuts'] == resolved,:]

data = data.drop(columns=['generation', 'cuts'])
data = data[data['LCOE'] < costConstraint*mincost]

data['solar'] = data[[f'var{n}' for n in range(1, pidx+1)]].sum(axis=1)
data['wind'] = data[[f'var{n}' for n in range(pidx+1, widx+1)]].sum(axis=1)
data['php'] = data[[f'var{n}' for n in range(widx+1, sidx+1)]].sum(axis=1)
data['phs'] = data[f'var{sidx+1}']

data['s/w'] = data['solar']/data['wind']
data['gen'] = data['solar'] + data['wind']
data['phhrs'] = data['phs']/data['php']

data = data.round(4)

#%% 

def compare_areas(data, val, cols, ticks, col_labels=None, threshold=None):
    if isinstance(ticks, int):
        ticks = np.linspace(1, threshold, ticks)
    col_labels = cols if col_labels is None else col_labels
    assert len(col_labels)==len(cols)
    ticks = np.array(ticks)
    ticks[::-1].sort()
    
    datamin = data[val].min()
    minima = pd.DataFrame([], index=ticks, columns=cols)
    maxima = pd.DataFrame([], index=ticks, columns=cols)
    d = data.copy()
    for thresh in ticks: 
        d = d[d[val]<datamin*thresh]
        minima.loc[thresh, cols] = d[cols].min()
        maxima.loc[thresh, cols] = d[cols].max()
    del d
    
    minima = minima.reset_index(names='threshold').astype(float)
    maxima = maxima.reset_index(names='threshold').astype(float)
        
    palette = sns.color_palette(n_colors=len(cols))
    
    fig, axs = plt.subplots(len(cols))
    
    for i, col in enumerate(cols):
        axs[i].fill_between(
            x = maxima['threshold'], 
            y1 = maxima[col], 
            y2 = minima[col], 
            color=palette[i],
            )
        
        if i < len(cols)-1:
            axs[i].set_xticks([])
        else: 
            axs[i].set_ylabel('Near-optimal threshold')
        
        # axs[i].set_ylim([0, None])
        axs[i].set_ylabel(col_labels[i])
        
    fig.suptitle("Ranges of total installed capacity at varying near-optimal thresholds")

compare_areas(
    data, 
    val='LCOE', 
    cols=['solar','wind','php','phs'],
    col_labels=['Solar (GW)', 'Wind (GW)','Storage (GW)','Storage (GWh)'],
    ticks=[1.01, 1.015, 1.02]#, 1.03, 1.04, 1.05, 1.075, 1.1, 1.15, 1.2]
    )
    
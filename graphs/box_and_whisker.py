# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:40:40 2024

@author: u6942852
"""

import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize

import graphutils as gu

scenario=21
dpi=250


os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
from Input import TSWind, TSPV, nodes
pidx, widx, sidx, headers = gu.zoneTypeIndx(scenario)




costConstraint=1.01

file =fr"Results\History{scenario}-children.csv"
data = pd.read_csv(file, header=None)


os.chdir('graphs')

solarCols=[f'var{n}_pv' for n in range(1, pidx+1)]
windCols=[f'var{n}_w' for n in range(pidx+1, widx+1)]
phpCols=[f'var{n}_php' for n in range(widx+1, sidx+1)]
phsCols=[f'var{sidx+1}_phs']

varCols=solarCols+windCols+phpCols+phsCols

data.columns = ['objective', 'generation', 'cuts', 'LCOE', 'LCOG', 'LCOBS', 'LCOBT', 'LCOBL']+varCols
# data.columns = ['LCOE', 'generation', 'cuts']+varCols

data['penalties'] = (data['objective']-data['LCOE']).round(5)
data = data[data['penalties'] <0.1]

mincost = data['LCOE'].min()
resolved = data['cuts'].max()

data = data.loc[data['cuts'] == resolved,:]

data = data.drop(columns=['generation', 'cuts'])
data = data[data['LCOE'] < costConstraint*mincost]

data['solar'] = data[solarCols].sum(axis=1)
data['wind'] = data[windCols].sum(axis=1)
data['php'] = data[phpCols].sum(axis=1)
data['phs'] = data[phsCols].sum(axis=1)

data['s/w'] = data['solar']/data['wind']
data['gen'] = data['solar'] + data['wind']
data['phhrs'] = data['phs']/data['php']

data = data.round(4)

#%%

def box_and_whisker_per_zone(data, ax=None, xscale=1.5, yscale=0.5):
    ax = ax if ax is not None else plt.gca()
    fig=plt.gcf()
    assert len(data)>0
    melted_data = data.melt(
        id_vars=[],
        value_vars=data.columns,
        var_name='var2',
        value_name='capacity',
        )
    melted_data['source'] = melted_data['var2'].str.extract(r'([a-zA-Z]+)')
    # melted_data['var1'] = melted_data['var']
    # melted_data[['var','source']] = melted_data['var'].str.split('-', n=1, expand=True)
    # melted_data['varn'] = melted_data['var'].str.extract(r'(\d+)').astype(int)
    # melted_data['var2']=''
    # melted_data.loc[melted_data['source'] == 'pv', 'var2'] = 'pv-' + melted_data['varn'][melted_data['source'] == 'pv'].astype(str)
    # melted_data.loc[melted_data['source'] == 'w', 'var2'] = 'w-' + (melted_data['varn'][melted_data['source'] == 'w'] - pidx).astype(str)
    # melted_data.loc[melted_data['source'] == 'php', 'var2'] = 'php-' + (melted_data['varn'][melted_data['source'] == 'php'] - sidx).astype(str)
    # melted_data.loc[melted_data['source'] == 'phs', 'var2'] = 'phs-1'
    melted_data['source']=melted_data['source'].apply(
        lambda x: {'pv':'Solar (GW)', 
                    'w':'Wind (GW)', 
                    'sp':'Storage (GW)', 
                    'se':'Storage (GWh)'}.get(x,x))
    
    sns.boxplot(
        melted_data[melted_data.loc[:,'source'] != 'Storage (GWh)'],
        x='var2',
        y='capacity',
        hue='source',
        hue_order=['Solar (GW)','Wind (GW)', 'Storage (GW)', 'Storage (GWh)'],
        ax=ax,
        )
    ax2=ax.twinx()
    sns.boxplot(
        melted_data[melted_data.loc[:,'source'] == 'Storage (GWh)'],
        x='var2',
        y='capacity',
        hue='source',
        hue_order=['Solar (GW)','Wind (GW)', 'Storage (GW)', 'Storage (GWh)'],
        ax=ax2,
        legend=False
        )
    
    ax.set_ylim(0,None)
    ax2.set_ylim(0,None)

    # ax.set_xticks([])
    ax.tick_params(axis='x', size=8)
    plt.setp(ax.get_xticklabels(), rotation=-90, ha='right')
    ax.set_ylabel("Installed Capacity (GW)")
    ax2.set_ylabel("Installed Capacity (GWh)")
    ax.set_xlabel("Zone")
    
    # gu.adjust_legend([ax, ax2], xscale, yscale) 
    # gu.adjust_legend([ax], xscale, yscale) 
    


# fig, axs = plt.subplots(2, dpi=dpi)
# box_and_whisker_per_zone(data, ax=axs[0])
# box_and_whisker_per_zone(data[data['s/w'] >= 0.75], ax=axs[1])
# fig.subplots_adjust(hspace=0.95)

# axs[0].set_title("All near-optimal")
# axs[1].set_title("Solar/Wind ratio at least 2")

d = data[varCols]
d.columns = ([f'pv-{n}' for n in range(1, pidx+1)] +
             [f'w-{n}' for n in range(1, widx-pidx+1)] +
             [f'sp-{n}' for n in range(1, sidx-widx+1)]+
             ['se-1'])

dpi = 1000
fig, axs = plt.subplots(2, 1, figsize = (10, 8), dpi=dpi, sharex=True)
fig.subplots_adjust(hspace=0.3)
box_and_whisker_per_zone(d, axs[0], 1.32, 0.5)
axs[0].set_title("a. Distribution of asset sizes for all near-optimal configurations")

cfs = np.concatenate((TSPV.mean(axis=0), TSWind.mean(axis=0), np.zeros(nodes+1)))
pvs = [f'pv-{n}' for n in range(1, pidx+1)]
ws = [f'w-{n}' for n in range(1, widx-pidx+1)]
labs=pvs+ws+[f'sp-{n}' for n in range(1, nodes+1)] + ['se-1']


axs[1].bar(pvs, TSPV.mean(axis=0), facecolor=sns.color_palette()[0])
axs[1].bar(ws, TSWind.mean(axis=0), facecolor=sns.color_palette()[1])
axs[1].scatter(labs, 0.1*np.zeros(len(labs)), color=[0,0,0,0])

# ax2=axs[1].twinx()
# ax2.scatter(labs, 0.1*np.zeros(len(labs)), color=[0,0,0,0])

axs[1].set_ylim(0,None)
axs[1].set_title("b. Average capacity factor of zones")
axs[1].set_xlabel("Zone")
axs[1].set_ylabel("Average capacity factor")
plt.setp(axs[1].get_xticklabels(), rotation=-90, ha='right')

# gu.adjust_legend(list(axs), 1.32, 0.5) 
plt.show()

#%% 

def box_and_whisker_power(data, ax=None, xscale=1.5, yscale=0.5):
    ax = ax if ax is not None else plt.gca()
    fig=plt.gcf()
    assert len(data)>0
    melted_data = data.melt(
        id_vars=[],
        value_vars=data.columns,
        var_name='var2',
        value_name='capacity',
        )
    melted_data['source'] = melted_data['var2'].str.extract(r'([a-zA-Z]+)')
    # melted_data['var1'] = melted_data['var']
    # melted_data[['var','source']] = melted_data['var'].str.split('-', n=1, expand=True)
    # melted_data['varn'] = melted_data['var'].str.extract(r'(\d+)').astype(int)
    # melted_data['var2']=''
    # melted_data.loc[melted_data['source'] == 'pv', 'var2'] = 'pv-' + melted_data['varn'][melted_data['source'] == 'pv'].astype(str)
    # melted_data.loc[melted_data['source'] == 'w', 'var2'] = 'w-' + (melted_data['varn'][melted_data['source'] == 'w'] - pidx).astype(str)
    # melted_data.loc[melted_data['source'] == 'php', 'var2'] = 'php-' + (melted_data['varn'][melted_data['source'] == 'php'] - sidx).astype(str)
    # melted_data.loc[melted_data['source'] == 'phs', 'var2'] = 'phs-1'
    melted_data['source']=melted_data['source'].apply(
        lambda x: {'pv':'Solar (GW)', 
                    'w':'Wind (GW)', 
                    'sp':'Storage (GW)', 
                    'se':'Storage (GWh)'}.get(x,x))
    
    melted_data['var2']=melted_data['var2'].apply(
        lambda x: {'sp-1':'NSW', 
                   'sp-2':'QLD', 
                   'sp-3':'SA', 
                   'sp-4':'TAS', 
                   'sp-5':'VIC'}.get(x,x))
    
    sns.boxplot(
        melted_data[melted_data.loc[:,'source'] == 'Storage (GW)'],
        x='var2',
        y='capacity',
        # hue='source',
        # hue_order=['Solar (GW)','Wind (GW)', 'Storage (GW)', 'Storage (GWh)'],
        ax=ax,
        legend=None,
        )
    
    ax.set_ylim(0,None)

    # ax.set_xticks([''])
    ax.set_ylabel("Installed Capacity (GW)")
    ax.set_xlabel("Zone")
    
    # gu.adjust_legend([ax, ax2], xscale, yscale) 
    # gu.adjust_legend([ax], xscale, yscale) 
    


d = data[varCols]
d.columns = ([f'pv-{n}' for n in range(1, pidx+1)] +
             [f'w-{n}' for n in range(1, widx-pidx+1)] +
             [f'sp-{n}' for n in range(1, sidx-widx+1)]+
             ['se-1'])

dpi = 1000
fig, axs = plt.subplots(2, 1, figsize = (10, 8), dpi=dpi, sharex=True)
box_and_whisker_power(d, axs[0])#, 1.32, 0.5)
box_and_whisker_power(d[data['s/w'] > data['s/w'].quantile(0.9)], axs[1])#, 1.32, 0.5)
axs[0].set_title("a. Distribution of storage power required for all near-optimal configurations")
axs[1].set_title("a. Distribution of storage power required for high-solar near-optimal configurations")


plt.show()



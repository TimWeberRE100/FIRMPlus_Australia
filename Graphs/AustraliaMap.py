# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:32:22 2024

@author: u6942852
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

cp = sns.color_palette()

if os.getcwd().split('\\')[-1] == 'Graphs':
    os.chdir('..')


def plot_grid_balancing(solution, state=None):
    state = 'system' if state is None else state
    # EGB = pd.read_csv(fr'Results/EGB{solution.scenario}.csv')
    fig, ax = plt.subplots(1, 2)
    # deficit-filling
    deficit = [EGB.loc[EGB['State'] == state, 'Storage Deficit Filling'         ].iloc[0],
               EGB.loc[EGB['State'] == state, 'Transmission Deficit Filling'    ].iloc[0],
               EGB.loc[EGB['State'] == state, 'Curtailment Deficit Filling'     ].iloc[0],
               EGB.loc[EGB['State'] == state, 'Diversity (pv) Deficit Filling'  ].iloc[0],
               EGB.loc[EGB['State'] == state, 'Diversity (onsw) Deficit Filling'].iloc[0],
               EGB.loc[EGB['State'] == state, 'Diversity (offw) Deficit Filling'].iloc[0],
               ]
    surplus = [EGB.loc[EGB['State'] == state, 'Storage Surplus Shaving'         ].iloc[0], 
               EGB.loc[EGB['State'] == state, 'Transmission Surplus Shaving'    ].iloc[0],
               EGB.loc[EGB['State'] == state, 'Curtailment Surplus Shaving'     ].iloc[0],
               EGB.loc[EGB['State'] == state, 'Diversity (pv) Surplus Shaving'  ].iloc[0],
               EGB.loc[EGB['State'] == state, 'Diversity (onsw) Surplus Shaving'].iloc[0],
               EGB.loc[EGB['State'] == state, 'Diversity (offw) Surplus Shaving'].iloc[0],
               ]
    labels = ['Storage', 'Transmission', 'Curtailment', 'Diversity (pv)', 
              'Diversity (onsw)', 'Diversity (offw)']
    def_to_surp = 1#sum(deficit)/sum(surplus)
    # print(def_to_surp)
    ax[0].pie(
        x = deficit,
        # colors = [cp[0], cp[1], cp[2], cp[3], cp[4], cp[5]],
        radius = def_to_surp,
        )
    ax[0].set_title('Deficit Filling')
    

    w = ax[1].pie(
        x = surplus,
        # colors = [cp[0], cp[1], cp[2], cp[3], cp[4], cp[5]],
        radius = def_to_surp**-1,
        )
    ax[1].set_title('Surplus Shaving')
    ax[1].legend(w[0], labels, loc = 'lower center', bbox_to_anchor=(0, -0.2, 0, -1.0), ncols=3)
    fig.suptitle(f'Grid Balancing - {state}')
    
EGB = pd.read_csv(r'Results/EGB21.csv')
for state in EGB['State'].unique():
    plot_grid_balancing(None, state)
raise KeyboardInterrupt

#%%
# from Input import pidx, widx, sidx, scenario, resolution, years, CHydro, Nodel
pidx, widx, sidx, scenario, resolution, years  = 5, 10, 15, 31, 0.5, 10
CHydro = np.array([2.419258, 0.1682  , 0.00321 , 2.2876  , 2.294337])
Nodel = np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC'])


CHYDRO = CHydro.copy()

scenario=31
# scenario='HighSolar31'

graphs = 'energy'
# graphs = 'power'
# graphs = 'both'

states = gpd.read_file('Data/states.geojson')
states = states.loc[states['STATE_NAME'] != 'Western Australia', :]
states = states.loc[states['STATE_NAME'] != 'Australian Capital Territory', :]
states = states.loc[states['STATE_NAME'] != 'Northern Territory', :]


rekey = {'1':'NSW', '2':'VIC', '3':'QLD', '4':'SA', '5':'WA','6':'TAS', '7':'NT', '8':'ACT'}
states_ = {rekey[c]:x for c, x in zip(states['STATE_CODE'], states['geometry'])}

nsw_xy = states_['NSW'].centroid.x, states_['NSW'].centroid.y
vic_xy = states_['VIC'].centroid.x, states_['VIC'].centroid.y
qld_xy = states_['QLD'].centroid.x, states_['QLD'].centroid.y
sa_xy  = states_['SA'].centroid.x, states_['SA'].centroid.y
tas_xy = states_['TAS'].centroid.x, states_['TAS'].centroid.y
xys = {'NSW':nsw_xy, 
       'VIC':vic_xy,
       'QLD':qld_xy,
       'SA':sa_xy, 
       'TAS':tas_xy,
       }

def draw_pie(dist, xpos, ypos, size, ax, fig, labels, colors):
    # for incremental pie slices
    cumsum = np.cumsum(dist)
    cumsum = cumsum/ cumsum[-1]
    pie = [0] + cumsum.tolist()

    for i, r in enumerate(zip(pie[:-1], pie[1:])):
        r1, r2 = r
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()

        xy = np.column_stack([x, y])
    
        ax.scatter(
            [xpos], 
            [ypos], 
            marker=xy, 
            s=size, 
            zorder=1000, 
            # label=labels[i], 
            color=colors[i]
            )
        ax.scatter(
            [xpos], 
            [ypos], 
            marker='o',
            s=size,
            zorder=1001,
            facecolor=[1,1,1,0],
            edgecolor=[0,0,0,1],
            linewidth=0.5
            )

    return ax

labels=['Solar PV', 'Wind', 'Hydro', 'PHES']
colors=[cp[1], cp[5], cp[0], cp[2]]

if graphs == 'both':
    fig, axs = plt.subplots(1, 2, figsize = (7,6), dpi=1600, sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.1)
    
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    states.plot(ax=axs[0], edgecolor='black', facecolor='grey', zorder=10)
    states.plot(ax=axs[1], edgecolor='black', facecolor='grey', zorder=10)
else: 
    fig, ax = plt.subplots(1, figsize = (3,6), dpi=1600)

    ax.set_xticks([])
    ax.set_yticks([])

    states.plot(ax=ax, edgecolor='black', facecolor='grey', zorder=10)

    # [ax.spines[pos].set_visible(False) for pos in ('left', 'right', 'top', 'bottom')]


# =============================================================================
# Energy Graph
# =============================================================================
if graphs == 'both':
    ax = axs[0]
    
if graphs == 'both' or graphs == 'energy':
    # labels=['Solar PV', 'Wind', 'Hydro']
    # colors=[cp[1], cp[5], cp[0]]
    
    sizes = {}
    # plot energy mix 
    for s in ('NSW', 'VIC', 'SA', 'TAS', 'QLD'):
        lpgm = pd.read_csv(f'Results/S{scenario}{s}.csv')
        
        GPV, GWind, GHydro, GPHES = (lpgm[col].sum().sum()*resolution/years/pow(10, 6) 
                                      for col in ('Solar photovoltaics', 'Wind', 
                                                  ['Hydropower', 'Biomass'], 
                                                  'Pumped hydro energy storage'))
        sf= 17
        draw_pie(np.array([GPV, GWind, GHydro]), 
                 *xys[s], 
                 size = sum([GPV, GWind, GHydro]) *sf,
                 ax=ax,
                 fig=fig,
                 labels=labels, 
                 colors=colors,
                 )
        draw_pie(np.array([GPHES]), 
                 *xys[s], 
                 size = GPHES *sf ,
                 ax=ax,
                 fig=fig,
                 labels=['PHES'], 
                 colors=[cp[2]],
                 )
    
        sizes[s] = sum([GPV, GWind, GHydro])
    
    
    
    lpgm = pd.read_csv(f'Results/S{scenario}.csv')
    scalefactor=6
    
    # hvdc lines
    for node_pair in ('NSW-VIC', 'NSW-QLD', 'NSW-SA', 'TAS-VIC'):
        forward=np.maximum(0, lpgm[node_pair]).sum()*resolution/years*pow(10,-6)
        reverse=np.minimum(0, lpgm[node_pair]).sum()*resolution/years*pow(10,-6)
        n1_xy, n2_xy = xys[node_pair[:3]], xys[node_pair[4:]]
        dxy1 = [(c2 - c1) for c1, c2 in zip(n1_xy, n2_xy)]
        dxy2 = [(c2 - c1) for c1, c2 in zip(n2_xy, n1_xy)]
        
        alph1 = np.arctan2(dxy1[1], dxy1[0])
        alph2 = np.arctan2(dxy2[1], dxy2[0])
    
        d1 = (dxy1[0]**2 + dxy1[1]**2)**0.5 - (sizes[node_pair[4:]]**0.5)/3
        d2 = (dxy2[0]**2 + dxy2[1]**2)**0.5 - (sizes[node_pair[:3]]**0.5)/3
        
        dxy1 = d1*np.cos(alph1), d1*np.sin(alph1)
        dxy2 = d2*np.cos(alph2), d2*np.sin(alph2)
        
        ax.arrow(
            *n1_xy, 
            *dxy1,
            color='red', 
            length_includes_head=True, 
            width=0.005,
            head_width=forward/scalefactor, 
            head_length=d1/5,
            zorder=10,
            head_starts_at_zero=True,
            )
        ax.arrow(
            *n2_xy,
            *dxy2,
            color='red', 
            length_includes_head=True, 
            width=0.005,
            head_width=reverse/scalefactor,
            head_length=d2/5,
            zorder=10,
            head_starts_at_zero=True,
            )
    
    ax.set_title('Energy Production and Transmission')

# =============================================================================
# Capacity graph
# =============================================================================
    
if graphs == 'both':
    ax = axs[1]
if graphs == 'both' or graphs == 'power':    
    # plot energy mix 
    if isinstance(scenario, int):
        x = pd.read_csv(f'Results/Optimisation_resultx{scenario}.csv', header=None).to_numpy().flatten()
    elif isinstance(scenario, str):
        x = pd.read_csv(f'Results/{scenario}.csv', header=None).to_numpy().flatten()
    for i, s in enumerate(('NSW', 'VIC', 'SA', 'TAS', 'QLD')):
        n = np.where(s==Nodel)[0][0]
        CPV = x[n]
        CWind = x[pidx+n]
        CPHES = x[widx+n]
        CHydro = CHYDRO[n]
        
        draw_pie(np.array([CPV, CWind, CPHES, CHydro]), 
                 *xys[s], 
                 size = sum([CPV, CWind, CPHES, CHydro]) * 28,
                 ax=ax,
                 fig=fig,
                 labels=labels, 
                 colors=colors,
                 )
    
    lpgm = pd.read_csv(f'Results/S{scenario}.csv')
    
    scalefactor=1100
    # hvdc lines
    for node_pair in ('NSW-VIC', 'NSW-QLD', 'NSW-SA', 'TAS-VIC'):
        ax.plot(*zip(xys[node_pair[:3]], xys[node_pair[4:]]), color='red', linewidth = np.abs(lpgm[node_pair]).max()/scalefactor, zorder=11)
    
    ax.set_title('Power Capacity')
    
if graphs=='both':
    d = [axs[1].scatter([xys[s][0]], [xys[s][1]], label=labels[i], color=colors[i], marker='s', zorder=0) for i in range(4)]
    pairs = dict(zip(labels, d))
    fig.legend(pairs.values(), pairs.keys(), bbox_to_anchor=(0.8, 0.18), ncols=4)
else:
    d = [ax.scatter([xys[s][0]], [xys[s][1]], label=labels[i], color=colors[i], marker='s', zorder=0) for i in range(4)]
    pairs = dict(zip(labels, d))
    fig.legend(pairs.values(), pairs.keys(), bbox_to_anchor=(0.9, 0.2), ncols=2)
   

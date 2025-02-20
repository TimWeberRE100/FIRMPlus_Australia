# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:40:56 2024

@author: u6942852
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as pltd
import datetime as dt
import seaborn as sns
import os 

def directory_up():
    os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))

directory_up()

from Input import * 

def plot_trace(S, start=None, end=None, sbplt_kwargs={}):
    if not S.evaluated:
        S._evaluate()
    
    datentime = np.array([(dt.datetime(firstyear, 1, 1, 0, 0) + x * dt.timedelta(minutes=60 * resolution)) for x in range(intervals)])
    
    pos_cols = ['Hydro & Bio (MW)', 'Solar (MW)', 'Wind (MW)', 'Storage (MW)', 'Transmission (MW)', 'Spillage (MW)']
    neg_cols = ['Transmission (MW)', 'Storage (MW)'] 
    
    plotarr = np.stack([arr[start:end] for arr in (S.flexible + S.GBaseload, S.GPV, S.GWind, S.Discharge, S.Import, S.Spillage)])
    
    for n in range(S.nodes):
        fig, ax = plt.subplots(**sbplt_kwargs)
        
        l = ax.stackplot(
            datentime[start:end],
            [arr[:,n] for arr in plotarr],
            labels = pos_cols,
            )
        l[0].set_color(sns.color_palette()[5]) # Hydro Bio
        l[1].set_color(sns.color_palette()[0]) # Solar
        l[2].set_color(sns.color_palette()[1]) # Wind 
        l[3].set_color(sns.color_palette()[2]) # Charge 
        l[4].set_color(sns.color_palette()[4]) # Import 
        l[5].set_color(sns.color_palette()[6]) # Spillage
        
        l = ax.stackplot(
            datentime[start:end], 
            (-S.Export[start:end, n], -S.Charge[start:end, n]), 
            labels = neg_cols,
            )
        l[0].set_color(sns.color_palette()[4]) # Export
        l[1].set_color(sns.color_palette()[2]) # Charge

        ax.plot( # fix extra ring of colour
            datentime[start:end], 
            plotarr.sum(axis=0)[:,n],
            color = 'white',
            )
        
        ax.plot( # fix extra ring of colour
            datentime[start:end], 
            -S.Export[start:end, n] - S.Charge[start:end, n],
            color='white',
            )

        ax.plot(
            datentime[start:end], 
            S.MLoad[start:end, n], 
            color='cyan', 
            linestyle='-',
            linewidth=1.25, 
            label='Demand (MW)',
            )
        
        print(np.where(S.Deficit[start:end, n]>0))
        d=datentime[np.where(S.Deficit[:, n]>0)][start:end]
        if len(d)>0: 
            print(d)
            ax.axvline(
                d,
                color='red', 
                # alpha=0.5,
                linewidth=0.5,
                )

        axt = ax.twinx()
        axt.plot(
            datentime[start:end], 
            S.Storage[start:end, n] ,#/ S.CPHS[n] / 10,
            color='black',
            linestyle='--',
            label='Storage reserve (%)',
            )
        
        ax.xaxis.set_major_formatter(pltd.DateFormatter('%H:%M'))
        plt.setp(ax.get_xticklabels(), rotation=-45, ha='left')

        ax.set_xlabel('Date and Time')
        ax.set_ylabel('Power (MW)')
        axt.set_ylabel('Storage Energy (%)')
        # axt.set_ylim(-5,105)
        
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width*0.9, pos.height])
        
        lns, labs = ax.get_legend_handles_labels()
        lns2, labs2 = axt.get_legend_handles_labels()
        
        leg_dict = {**dict(zip(labs, lns)), ** dict(zip(labs2, lns2))}
        
        ax.legend(
            leg_dict.values()
            , leg_dict.keys()
            , loc = 'center right'
            , bbox_to_anchor = (1.64, 0.5)
            )

        ax.set_title(f'{Nodel[n]} - {datentime[start]} to {datentime[end]}')

        plt.show()

if __name__=='__main__':
    y = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)

    x1 = y.copy()
    x1[26] = 25.
    # plot_trace(Solution(y), 8950, 9050, {'dpi':500})
    plot_trace(Solution(x1), 9000, 9050, {'dpi':500})


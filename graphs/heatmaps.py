# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:44:01 2024

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
costConstraint=1.01



os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))

pidx, widx, sidx, headers = gu.zoneTypeIndx(scenario)
file =fr"Results\History{scenario}-children.csv"
data = pd.read_csv(file, header=None)

os.chdir('graphs')

#%%
varCols=[f'var{n}' for n in range(1, sidx+2)]
# data.columns = ['LCOE', 'generation', 'cuts']+varCols
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

# def costmap(data, x, y, val, colormap='rocket', reverse_color=False, ax=None, 
#             fig=None, x_bins='max', y_bins='max'):
#     ax = plt.gca() if ax is None else ax
#     fig = plt.gcf() if fig is None else fig
    
#     colormap = colormap+'_r' if reverse_color else colormap

#     xmin, xmax = data[x].min(), data[x].max()
#     ymin, ymax = data[y].min(), data[y].max()
    
#     x_bins = data[x].nunique() if x_bins=='max' else x_bins
#     y_bins = data[y].nunique() if y_bins=='max' else y_bins
    
#     Z = data.reset_index().pivot(index=y, columns=x, values=val).to_numpy()
    
#     X, Y = np.meshgrid(
#         np.linspace(xmin, xmax, x_bins),
#         np.linspace(ymin, ymax, y_bins))
    
#     c = ax.pcolormesh(X, Y, np.ma.masked_invalid(Z), cmap=colormap)
#     fig.colorbar(c, ax=ax)
#     ax.set_xlabel(x)
#     ax.set_ylabel(y)
#     ax.set_title(f"{val} by {x} and {y}")

#%%

def aggregate_data(data, x, y, val, agg):
    if agg == 'min': 
        return data.groupby([x, y])[val].min()
    if agg == 'max': 
        return data.groupby([x, y])[val].max()
    if agg == 'count': 
        return data.groupby([x, y])[val].count()
    if agg == 'mean': 
        return data.groupby([x, y])[val].mean()
    if agg == 'median': 
        return data.groupby([x, y])[val].median()
    raise Exception

def continuous_heatmap(data, x, y, val, agg='min', colormap='rocket', reverse_color=False, ax=None, 
                        fig=None, x_bins='max', y_bins='max'):
    assert agg in ('min', 'max', 'count', 'mean','median')
    ax = plt.gca() if ax is None else ax
    fig = plt.gcf() if fig is None else fig
    
    colormap = colormap+'_r' if reverse_color else colormap

    xmin, xmax = data[x].min(), data[x].max()
    ymin, ymax = data[y].min(), data[y].max()
    
    x_bins = data[x].nunique() if x_bins=='max' else x_bins
    y_bins = data[y].nunique() if y_bins=='max' else y_bins

    data = aggregate_data(data, x, y, val, agg)
    
    Z = data.reset_index().pivot(index=y, columns=x, values=val).to_numpy()
    
    X, Y = np.meshgrid(
        np.linspace(xmin, xmax, x_bins),
        np.linspace(ymin, ymax, y_bins))
    
    c = ax.pcolormesh(X, Y, np.ma.masked_invalid(Z), cmap=colormap)
    fig.colorbar(c, ax=ax)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{agg} of {val} by {x} and {y}")
    

def compare_heatmaps(data, xs, ys, vals, aggs=['min'], colormap='rocket', reverse_color=False, 
                     axs=None, fig=None, x_bins=['max'], y_bins=['max'], share_cmap=True):
    fig = plt.gcf() if fig is None else fig
    assert axs is not None    
    axs=axs.flatten()
    
    for arg in (xs, ys, vals, aggs, x_bins, y_bins):
        assert isinstance(arg, (list, tuple))
    for agg in aggs: 
        assert agg in ('min', 'max', 'count', 'mean', 'median')
        
    lens = [len(arg) for arg in (xs, ys, vals, aggs, x_bins, y_bins)]
    maxlen = max(lens)
    
    if len(xs) == 1: xs=xs*maxlen
    if len(ys) == 1: ys=ys*maxlen
    if len(vals) == 1: vals=vals*maxlen
    if len(aggs) == 1: aggs=aggs*maxlen
    if len(x_bins) == 1: x_bins=x_bins*maxlen
    if len(y_bins) == 1: y_bins=y_bins*maxlen
    
    assert len(axs)>=maxlen
    
    colormap = colormap+'_r' if reverse_color else colormap

    x_bins = [data[x].nunique() if x_bins[i]=='max' else x_bins[i] for i, x in enumerate(xs)]
    y_bins = [data[y].nunique() if y_bins[i]=='max' else y_bins[i] for i, y in enumerate(ys)]
    
    Zs = [aggregate_data(data, xs[i], ys[i], vals[i], aggs[i])
          .reset_index().pivot(index=ys[i], columns=xs[i], values=vals[i])
          .to_numpy() for i in range(maxlen)]
    
    if share_cmap is True:
        Zmin, Zmax = min(np.nanmin(Z) for Z in Zs), max(np.nanmax(Z) for Z in Zs)
        norm = Normalize(Zmin, Zmax)
    else: 
        norm=None
    
    for i in range(maxlen):
        X, Y = np.meshgrid(
            np.linspace(data[xs[i]].min(), data[xs[i]].max(), x_bins[i]), 
            np.linspace(data[ys[i]].min(), data[ys[i]].max(), y_bins[i]))
        
        c = axs[i].pcolormesh(X, Y, np.ma.masked_invalid(Zs[i]), cmap=colormap, norm=norm)
        
        if share_cmap is False:
            fig.colorbar(c, ax=axs[i])

        axs[i].set_xlabel(xs[i])
        axs[i].set_ylabel(ys[i])
        axs[i].set_title(f"{aggs[i]} of {vals[i]} by {xs[i]} and {ys[i]}")
    
    if share_cmap is True:
        fig.colorbar(c, ax=axs.ravel().tolist())
        
def heatmap_pairplot(data, val, vars=None, x_vars=None, y_vars=None, agg='min',
                     colormap='rocket_r', x_bins=['max'], y_bins=['max'], 
                     share_cmap=True, subplot_kwargs={}, figaxs=None):
    xs = vars if vars is not None else x_vars if x_vars is not None else data.columns
    ys = vars if vars is not None else y_vars if y_vars is not None else data.columns
    
    fig, axs =figaxs if figaxs is not None else plt.subplots(len(ys), len(xs), **subplot_kwargs)
    

    maxlen = [len(arg) for arg in (xs, ys, x_bins, y_bins)]
    maxlen = max(maxlen)
    
    if len(x_bins) == 1: x_bins=x_bins*maxlen
    if len(y_bins) == 1: y_bins=y_bins*maxlen

    x_bins = [data[x].nunique() if x_bins[i]=='max' else x_bins[i] for i, x in enumerate(xs)]
    y_bins = [data[y].nunique() if y_bins[i]=='max' else y_bins[i] for i, y in enumerate(ys)]

    Zs = [aggregate_data(data, xs[col], ys[row], val, agg).reset_index()
          .pivot(index=ys[row], columns=xs[col], values = val).to_numpy()          
          for row in range(len(ys)) for col in range(len(xs)) if row != col ]
    
    if share_cmap is True:
        Zmin, Zmax = min(np.nanmin(Z) for Z in Zs), max(np.nanmax(Z) for Z in Zs)
        norm = Normalize(Zmin, Zmax)
    else: 
        norm=None

    k=0
    for row in range(len(ys)):
        for col in range(len(xs)):
            
            fontsize=10+2*len(ys)
            axs[row, col].tick_params(labelsize=fontsize)
            
            if row == len(ys)-1:
                axs[row, col].set_xlabel(xs[col], fontsize=fontsize)
            else: 
                axs[row, col].set_xticklabels([])
            if col == 0:
                axs[row, col].set_ylabel(ys[row], fontsize=fontsize)
            else: 
                axs[row, col].set_yticklabels([])
            if row == col: 
# =============================================================================
#                 diagonal plots
# =============================================================================
                axs[row, col].hist(
                    data[xs[col]], 
                    bins=x_bins[col], 
                    )
                axs[row, col].set_ylim([None, None])
                axs[row, col].set_xlim([None, None])
                axs[row, col].set_yticks([])
                
                
                
# =============================================================================
#                 upper triangle
# =============================================================================
            if row < col:             
                # axs[row, col].set_ylim([data[ys[row]].min()*0.98, data[ys[row]].max()*1.02])
                # axs[row, col].set_xlim([data[xs[col]].min()*0.98, data[xs[col]].max()*1.02])

                # X, Y = np.meshgrid(
                #     np.linspace(data[xs[col]].min(), data[xs[col]].max(), x_bins[col]), 
                #     np.linspace(data[ys[row]].min(), data[ys[row]].max(), y_bins[row]))
                
                # # c_upper = axs[row,col].pcolormesh(X, Y, np.ma.masked_invalid(Zs[k]), cmap=colormap, norm=norm)
                # c = axs[row,col].pcolormesh(X, Y, np.ma.masked_invalid(Zs[k]), cmap=colormap, norm=norm)
                k+=1
            if row > col: 
                axs[row, col].set_ylim([data[ys[row]].min()*0.98, data[ys[row]].max()*1.02])
                axs[row, col].set_xlim([data[xs[col]].min()*0.98, data[xs[col]].max()*1.02])
                
                X, Y = np.meshgrid(
                    np.linspace(data[xs[col]].min(), data[xs[col]].max(), x_bins[col]), 
                    np.linspace(data[ys[row]].min(), data[ys[row]].max(), y_bins[row]))
                
                # c_lower = axs[row,col].pcolormesh(X, Y, np.ma.masked_invalid(Zs[k]), cmap='viridis', norm=norm)
                c = axs[row,col].pcolormesh(X, Y, np.ma.masked_invalid(Zs[k]), cmap=colormap, norm=norm)
                k+=1
                pass
# =============================================================================
#                 lower triangle
# =============================================================================
            
            if share_cmap is False:
               cbar =  fig.colorbar(c, ax=axs[row, col])
               cbar.ax.tick_params(labelsize=fontsize)

            # k+=1
            
    
    if share_cmap is True:
    #     cbar = fig.colorbar(c_upper, ax=axs.ravel().tolist())
    #     cbar.ax.tick_params(labelsize=fontsize)
        
    #     cbar = fig.colorbar(c_lower, ax=axs.ravel().tolist())
    #     cbar.ax.tick_params(labelsize=fontsize)
        cbar = fig.colorbar(c, ax=axs.ravel().tolist())
        cbar.ax.tick_params(labelsize=fontsize)
    
    
    w, h = fig.get_size_inches()
    fontsize=10+4*len(ys)
    text_height = 4*fontsize #in point 
    yh=1-(text_height/72)/h # 1 pt = 1/72 inches
    if vars is None:
        vars = xs + [y for y in ys if y not in xs]
    
    fig.suptitle(f'{agg} of {val} by {", ".join(vars)}', 
                  fontsize=fontsize,
                  y=yh
                 )


    
    return fig, axs
    
#%%
fig, axs = plt.subplots(3, 2, figsize=(8, 9), dpi=250, sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.7, hspace=0.3)
compare_heatmaps(
    data, 
    ['phs'], 
    ['php'], 
    ['solar', 'wind', 'solar', 'wind', 'solar', 'wind'], 
    ['min', 'min', 'mean', 'mean', 'max', 'max'], 
    reverse_color=True,
    axs=axs, 
    fig=fig)

fig, axs = plt.subplots(1, 3, figsize=(12, 5), dpi=250, sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.7, hspace=0.3)
compare_heatmaps(
    data, 
    ['solar'], 
    ['wind'], 
    ['LCOE'], 
    ['min', 'median', 'count'], 
    reverse_color=True,
    axs=axs, 
    fig=fig, 
    share_cmap=False)


fig, axs = plt.subplots(4, 4, figsize=(18, 17), dpi=500, sharex=False, sharey=False)
heatmap_pairplot(data, 'LCOE', vars=['solar', 'wind', 'php', 'phs'], figaxs=(fig,axs))


#%%
fig, axs = plt.subplots(1, 2, figsize=(8, 5), dpi=250, sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.7, hspace=0.3)
compare_heatmaps(
    data, 
    ['solar'], 
    ['gen'], 
    ['LCOE'], 
    ['min', 'count'], 
    reverse_color=True,
    axs=axs, 
    fig=fig, 
    share_cmap=False)
        
# fig, ax = plt.subplots()
# continuous_heatmap(data, 'gen', 'phs', 'LCOE', 'min', reverse_color=True, ax=ax, fig=fig)
            
# fig, ax = plt.subplots()
# continuous_heatmap(data, 'php', 'phs', 'LCOE', 'min', reverse_color=True, ax=ax, fig=fig)

plt.show()

#%%

from numba import njit, objmode

@njit
def is_pareto_efficient(arr):
    is_efficient = np.arange(arr.shape[0])
    n_points = arr.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(arr):
        nondominated_point_mask = np.empty(len(arr), np.bool_)
        with objmode(nondominated_point_mask='boolean[:]'):
            nondominated_point_mask = np.any(arr<arr[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        arr = arr[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1

    is_efficient_mask = np.zeros(n_points, dtype = np.bool_)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask

def pareto_points(data, cols, minimise=True, precision=0.001):
    if isinstance(minimise, (list, tuple)):
        assert len(minimise) == len(cols)

    d = data[cols].round(int(-np.log10(precision)))
    d = d.sort_values(cols, ascending=minimise)
    dindex = d.index

    mask = pd.Series(is_pareto_efficient(d.to_numpy()), index = dindex)
    
    d = d[mask].reset_index()['index']
    
    d = pd.merge(data.reset_index(), d, on = 'index', how='inner')
    d.index = d['index']
    d=d.drop(columns='index')
    
    return d
    

def pareto_pairplot(data, cols, plotcols=None):
    """ convenience wrapper to plot in the style I like managing title etc. """
    
    paretodata = pareto_points(data, cols)
    plotcols = cols if plotcols is None else plotcols
    
    g = sns.PairGrid(paretodata[plotcols])

    g.map_diag(sns.kdeplot, fill=True, common_norm=False)
    g.map_lower(sns.scatterplot)
    # g = sns.pairplot(
    #     paretodata[plotcols],
    #     diag_kind='kde',
    #     )
    # g.map_upper(sns.kdeplot, levels=max(4,len(paretodata)//15), cmap='rocket')#, fill=True, thresh=0)
    g.data=data
    g.map_upper(sns.histplot,  cmap='rocket')
    
    w, h = plt.gcf().get_size_inches()
    fontsize=10+4*len(plotcols)
    text_height = 2.02*fontsize #in point + 0.2 for linebreak
    yh=1+(text_height/72)/h # 1 pt = 1/72 inches
    plt.suptitle('near-optimal pareto efficient networks\npareto:'+'-'.join(cols), 
                 fontsize=fontsize,
                 y=yh)

# cols=['wind','solar','php','phs']
# pareto_pairplot(
#     data, 
#     cols, 
#     ['LCOE'] + cols)

# cols=['gen', 'php', 'phs']
# pareto_pairplot(
#     data, 
#     cols, 
#     ['LCOE', 's/w'] + cols)

pareto_pairplot(
    data, 
    ['LCOE', 'gen'])


pareto_pairplot(
    data, 
    ['LCOE', 'wind'],
    plotcols=['LCOE', 'solar', 'wind', 'phs'])

pareto_pairplot(
    data, 
    ['LCOE', 'phs'])

pareto_pairplot(
    data, 
    ['LCOE', 'php'])

# pareto_pairplot(
#     data, 
#     ['LCOE', 'gen', 'php'],
#     ['LCOE', 'wind', 'solar', 'php', 'phs'])



plt.show()

#%% 
from matplotlib import cm

def advanced_pareto_pairplot(data, cols, plotcols=None, fig=None, dpi=500):
    paretodata = pareto_points(data, cols)
    plotcols = cols if plotcols is None else plotcols
    
    g = sns.PairGrid(data[plotcols], corner=True, despine=False)
    fig = fig if fig is not None else plt.gcf()


    g.diag_sharey = False

    # x_bins = {x:data[x].nunique() for x in plotcols}
    # y_bins = {y:data[y].nunique() for y in plotcols}
    # bins = {**x_bins, **y_bins}
    bins={'LCOE': 20, 'Solar PV (GW)': 5, 'Wind (GW)': 3, 'Storage (GW)': 4, 
          'Wind to Solar ratio':15}
    
    def bivariate_hist(x, y, **kwargs):
        sns.histplot(
            x=x, 
            y=y, 
            bins=(bins[x.name], bins[y.name]), 
            cmap = 'copper_r',
            )
        
    def kdeplot_unsharedy(x, **kwargs):
        ax = plt.gca()
        
        sns.despine(ax=ax, left=True, top=True, right=False)
        ax.yaxis.tick_right()
        ax.set_ylabel('Density')
        
        sns.kdeplot(x=x, ax=ax, **kwargs)
        
    maxsize = 0
    for k, v in bins.items():
        while bins[k] > 512: 
            bins[k] = int(bins[k]/2) 
        
        slide = (data[k].max()-data[k].min())/ bins[k]
        if slide ==0:
            continue
        for edge in range(int(10000*data[k].min()), int(10000*data[k].max()), int(10000*slide)):
            size = data[k][(data[k] >= edge/10000) & (data[k] < (edge+slide)/10000)].count()
            maxsize = max(size, maxsize)
    g.data = data[plotcols]
    g.map_lower(bivariate_hist)
    g._despine=False
    g.map_diag(sns.kdeplot, color='orange', fill=True, common_norm=True, bw_adjust=5)

    g.data=paretodata[plotcols]
    # g.map_diag(sns.kdeplot, fill=True, common_norm=True)
    g.map_lower(sns.scatterplot, marker='x', linewidths = 1.8)
    
    for i, (y_var) in enumerate(g.y_vars):
        for j, (x_var) in enumerate(g.x_vars):
            # print(i,j,y_var, x_var)
            if i != j:
                sns.despine(ax=g.axes[i,j])
            # if i == j:
            #     sns.despine(ax=g.axes[i,j], left=True, right=False)
    # print(g.diag_axes)
    # print(g.offdiag_axes)
                

    cmap = cm.ScalarMappable(norm=Normalize(0, maxsize), cmap='copper_r')
    
    cbar = fig.colorbar(cmap, ax=[ax for ax in g.axes.ravel() if ax is not None])
    cbar.ax.set_ylim([cbar.norm(0), None])    
    cbar.ax.set_yticks([])

    fig.set_dpi(dpi)
    w, h = plt.gcf().get_size_inches()
    fontsize=10+4*len(plotcols)
    text_height = 3.02*fontsize #in point + 0.2 for linebreak
    yh=1+(text_height/72)/h # 1 pt = 1/72 inches
    plt.suptitle('Distribution of asset sizes in near-optimal networks (orange)\n'+
                 'and pareto efficient networks (blue) minimising\n'+', '.join(cols),
                 fontsize=fontsize,
                 y=yh)
    # print(g.axes)
    
if 'w/s' not in data.columns and 'Wind to Solar ratio' not in data.columns: 
    data['w/s'] = 1/data['s/w']
data = data.rename(columns={
    'wind':'Wind (GW)', 
    'solar':'Solar PV (GW)', 
    'php':'Storage (GW)',
    'w/s':'Wind to Solar ratio',
    'phs':'Storage (GWh)'})
# fig, ax = plt.subplots(dpi=500)
advanced_pareto_pairplot(
    data, 
    ['LCOE', 'Wind to Solar ratio'], 
    ['LCOE', 'Solar PV (GW)', 'Wind (GW)', 'Storage (GW)'],
    dpi=1500)

    
plt.show()

#%%
par = pareto_points(data, ['LCOE', 'Wind to Solar ratio'])
par = par.rename(columns={
    'gen':'Total installed generation (GW)',
    })

par = par.sort_values('Wind to Solar ratio')

fig, ax = plt.subplots(dpi=500)



ax.plot(par['Wind to Solar ratio'], 
        par['Total installed generation (GW)'],
        label = 'Total installed generation (GW)',
        color = sns.color_palette()[0],
        )

ax.plot(par['Wind to Solar ratio'], 
        par['Storage (GW)'],
        label = 'Storage (GW)',
        color = sns.color_palette()[1],
        )

ax2 = ax.twinx()
ax2.plot(par['Wind to Solar ratio'], 
        par['Storage (GWh)'],
        label = 'Storage (GWh)',
        color = sns.color_palette()[2],
        )

lns, labs = zip(*((items for items in x.get_legend_handles_labels()) for x in (ax, ax2)))
lns = [l for ln in lns for l in ln]
labs = [l for lab in labs for l in lab]
pairs = dict(zip(labs, lns))
ax.legend(pairs.values(), pairs.keys(), bbox_to_anchor=(0.7, 0.6)) 


ax.set_title("Wind to Solar ratio against selected attributes")
ax.set_ylabel("Capacity (GW)")
ax2.set_ylabel("Capacity (GWh)")
ax.set_xlabel("Wind to solar ratio (GW/GW)")


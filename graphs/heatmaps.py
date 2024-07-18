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

os.chdir('\\'.join(os.getcwd().split('\\')[:-1]))
from Input import scenario, lb, ub, pzones, wzones, nodes, pidx, widx, sidx

costConstraint=1.10

file =fr"Results\History{scenario}-children.csv"
data = pd.read_csv(file, header=None)

# f2 =fr"Results\History{scenario}-resolved.csv"
# data2 = pd.read_csv(f2, header=None)
# data = pd.concat((data, data2))

os.chdir('graphs')

#%%
varCols=[f'var{n}' for n in range(1, pzones+wzones+nodes+2)]
data.columns = ['cost', 'generation', 'cuts']+varCols

mincost = data['cost'].min()
resolved = data['cuts'].max()

fulldata = data.copy()
data = data.loc[data['cuts'] == resolved,:]

data = data.drop(columns=['generation', 'cuts'])
data = data[data['cost'] < costConstraint*mincost]

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
    ['cost'], 
    ['min', 'median', 'count'], 
    reverse_color=True,
    axs=axs, 
    fig=fig, 
    share_cmap=False)
        
# fig, ax = plt.subplots()
# continuous_heatmap(data, 'gen', 'phs', 'cost', 'min', reverse_color=True, ax=ax, fig=fig)
            
# fig, ax = plt.subplots()
# continuous_heatmap(data, 'php', 'phs', 'cost', 'min', reverse_color=True, ax=ax, fig=fig)

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
    
    g = sns.pairplot(
        paretodata[plotcols],
        diag_kind='kde',
        )
    g.map_upper(sns.kdeplot, levels=max(4,len(paretodata)//15), cmap='rocket')#, fill=True, thresh=0)
    
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
#     ['cost'] + cols)

# cols=['gen', 'php', 'phs']
# pareto_pairplot(
#     data, 
#     cols, 
#     ['cost', 's/w'] + cols)

# pareto_pairplot(
#     data, 
#     ['cost', 'gen'])

# pareto_pairplot(
#     data, 
#     ['cost', 'solar'])

# pareto_pairplot(
#     data, 
#     ['cost', 'phs'])

# pareto_pairplot(
#     data, 
#     ['cost', 'php'])

pareto_pairplot(
    data, 
    ['cost', 'gen', 'php'],
    ['cost', 'wind', 'solar', 'php', 'phs'])



plt.show()
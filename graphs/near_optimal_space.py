# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:07:20 2024

@author: u6942852
"""

import matplotlib.pyplot as plt 
import numpy as np

plt.rcParams.update({'mathtext.default':'regular'})

#%% Convex optimal space
fig, ax = plt.subplots(dpi=1500)

# Draw feasible space
ax.fill_between(
    [3,2,6,8,9,6,3],
    [1,4,5,5,3,0.5,1], 
    facecolor=[0,0,0,0.2],
    label='feasible space'
    )

# Draw optimal isoline
ax.plot(
    [1,4],
    [3,0],
    color=[0,0,0,1],
    linestyle=':',
    label='optimal isoline'
    )

#Draw near-optimal isoline
ax.plot(
    [1,6],
    [5,0],
    color=[0,0,0,0.7],
    linestyle='--',
    label='near-optimal isoline',
    )

ax.fill_between(
    [2,5.4,3,2],
    [4,0.6,1,4],
    facecolor=[1,0,0,0.5],
    label='near-optimal space',    
    )

# Draw optimum
ax.scatter(
    [3],
    [1], 
    marker ='*',
    facecolor=[1,0,0,1],
    edgecolor=[0,0,0,1],
    s=100,
    label='optimum',
    zorder=np.inf,
    )

ax.legend()
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('$x_2$')
ax.set_xlabel('$x_1$')

#%% Non-Convex Optimal Sapce

fig, ax = plt.subplots(dpi=1500)

# Draw feasible space
ax.fill_between(
    [3,4, 2, 2,6,8,9,6,3],
    [1,3, 3, 4,5,5,3,0.5,1], 
    facecolor=[0,0,0,0.2],
    label='feasible space'
    )

# Draw optimal isoline
ax.plot(
    [1,4],
    [3,0],
    color=[0,0,0,1],
    linestyle=':',
    label='optimal isoline'
    )

#Draw near-optimal isoline
ax.plot(
    [1,6],
    [5,0],
    color=[0,0,0,0.7],
    linestyle='--',
    label='near-optimal isoline',
    zorder=10000,
    )


ax.fill_between(
    [2,2,3,2],
    [3,4,3,3], 
    facecolor=[1,0,0,0.5],
    label='near-optimal space',    
    )

ax.fill_between(
    [3,3.65,5.4,3],
    [1,2.35,0.6,1], 
    facecolor=[1,0,0,0.5],
    label='near-optimal space',    
    )

# Draw optimum
ax.scatter(
    [3],
    [1], 
    marker ='*',
    facecolor=[1,0,0,1],
    edgecolor=[0,0,0,1],
    s=100,
    label='optimum',
    zorder=np.inf,
    )

# Include a hole 
ax.fill_between(
    [4,5.5,4,4],
    [2,2,1.5,2],
    facecolor=[1,1,1,1],
    zorder=1000
    )

handles, labels = plt.gca().get_legend_handles_labels()
by_label=dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('$x_2$')
ax.set_xlabel('$x_1$')

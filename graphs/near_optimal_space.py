# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:07:20 2024

@author: u6942852
"""

import matplotlib.pyplot as plt 
from matplotlib import patches
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

# Draw near-optimal space
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

#%% Convex Optimal Space

fig, ax = plt.subplots(dpi=1500)

xcenter, ycenter = 0.38, 0.52
width, height = 0.5, 1
angle = -30

theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
x = 0.5 * width * np.cos(theta)
y = 0.5 * height * np.sin(theta)

rtheta = np.radians(angle)
R = np.array([
    [np.cos(rtheta), -np.sin(rtheta)],
    [np.sin(rtheta),  np.cos(rtheta)],
    ])

x, y = np.dot(R, [x, y])
x += xcenter
y += ycenter

# ax.fill(
#     x, 
#     y, 
#     facecolor=[0,0,0,0.2],
#     linewidth=1, 
#     zorder=1
#     )

#feasible space
e1 = patches.Ellipse(
    (xcenter, ycenter), 
    width, 
    height, 
    angle=angle,
    linewidth=2,
    fill=True, 
    zorder=2,
    facecolor=[0,0,0,0.2],
    label='feasible space',
    )


#near-optimal space
e2 = patches.Ellipse(
    (xcenter*0.58, ycenter*0.58), 
    width*0.5, 
    height*0.4, 
    angle=angle+0.02,
    linewidth=2,
    fill=True, 
    zorder=3,
    facecolor=[1,0,0,0.5],
    label='near-optimal space',
    )

#near-optimal space
e3 = patches.Ellipse(
    (xcenter*0.58, ycenter*0.58), 
    width*0.5, 
    height*0.4, 
    angle=angle+0.02,
    linewidth=2,
    fill=False, 
    zorder=4,
    color=[0,0,0,0.7],
    label='near-optimal isoline',
    linestyle='--',
    )

# Draw optimum
ax.scatter(
    [0.1],
    [0.25], 
    marker ='*',
    facecolor=[1,0,0,1],
    edgecolor=[0,0,0,1],
    s=100,
    label='optimum',
    zorder=np.inf,
    )


ax.add_patch(e1)
ax.add_patch(e2)
ax.add_patch(e3)
# ax.add_patch(e4)


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
    label='global optimum',
    zorder=np.inf,
    )

# Draw optimum
ax.scatter(
    [2],
    [3], 
    marker ='*',
    facecolor=[1,0,0,0.9],
    edgecolor=[0,0,0,1],
    s=70,
    label='local optimum',
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


#%% 

fig, ax = plt.subplots()

x=np.linspace(0, 8, 200)
ax.plot(
    x,
    np.sin(2*np.pi*x)+ np.sin(1.5*np.pi*x))

ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('cost')
ax.set_xlabel('$x_1$')

#%% 

fig, ax = plt.subplots()

x=np.linspace(0, 8, 200)
ax.plot(
    x,
    ((x-3)**2)/5,
    )

ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('cost')
ax.set_xlabel('$x_1$')
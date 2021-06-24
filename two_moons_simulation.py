#!/bin/python

import MBO
import graph_laplacian
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n = 1000
N = 5

def eta_func(t):
    if t == 0:
        return 0

    return np.exp(-t/0.05) #the division by 0.05 is used to obtain very low weights for distant points

points, classes = make_moons(n, noise = 0.1)

L, W, deg = graph_laplacian.graph_Laplacian(points, eta_func)

km = KMeans(n_clusters = 2)
km.fit(points)

u = km.labels_

updates_u = np.zeros((n, N))

updates_u[:,0] = np.copy(u)

for x in range(N-1):
    u = MBO.MBO(L, u, mz=True, h = 40)
    updates_u[:,x+1] = np.copy(u)

def update_plot(i, data, scat):
    scat.set_array(data[:,i])
    return scat


fig = plt.figure()
scat = plt.scatter(points[:,0], points[:,1], c = u)

ani = animation.FuncAnimation(fig, update_plot, frames = N, fargs = (updates_u, scat), interval = 400)

writer = animation.FFMpegWriter(fps = 0.8)
ani.save("two_moons_sim.mp4", writer = writer)

plt.show()






# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 23:46:42 2019

@author: zhenh
"""

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")

np.random.seed(0)

centers = [[1,1,1],[5,5,5],[5,10,10]]

X, _ = make_blobs(n_samples = 200, centers = centers, cluster_std = 1)

ms = MeanShift()
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters:", n_clusters_)

colors = 10*['r','g','b','c','k','y','m']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

for cc in range(len(cluster_centers)):
    ax.scatter(cluster_centers[cc,0],cluster_centers[cc,1],cluster_centers[cc,2],
            marker="x",c=colors[cc], s=150, linewidths = 20, zorder=10)

plt.show()
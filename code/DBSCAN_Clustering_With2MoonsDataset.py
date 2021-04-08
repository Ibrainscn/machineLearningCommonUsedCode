# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 01:26:39 2019

@author: zhenh
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

plt.style.use('ggplot')

X, label = make_moons(n_samples=200, noise=0.1, random_state=19)

# model = DBSCAN?
model = DBSCAN(eps=0.25, min_samples=9).fit(X)

# Plot the clusters in feature space
fig, ax = plt.subplots(figsize=(10, 8))
sctr = ax.scatter(X[:, 0], X[:, 1], c=model.labels_, s=140, alpha=0.9,
                 cmap=plt.cm.Accent)
fig.show()

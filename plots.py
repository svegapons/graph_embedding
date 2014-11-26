# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 15:39:17 2014

@author: jm
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pandas import *

# %%
def heatmap(data):
    """Print a heatmap of the data
    for better understanding the range
    of values"""
    dim = np.shape(data)
    plt.figure('Heatmap of original data: X')
    if (len(dim) > 2):
        for i in range(len(data)):
            plt.subplot(int(np.ceil(dim[0]/6)), 6, i+1)
            plt.imshow(data[i])
            plt.colorbar()
            plt.hold(True)
            plt.axis('tight')
    else:
        plt.imshow(data)
        plt.colorbar()
        plt.axis('tight')
    plt.show()


# %%
def hinton(data, max_weight=None, ax=None):
    """
    Hinton diagrams are useful for visualizing the values of a 2D array (e.g.
    a weight matrix): Positive and negative values are represented by white and
    black squares, respectively, and the size of each square represents the
    magnitude of each value.
    """
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(data).max())/np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(data):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    plt.show()

# %%
def scatter_mat(data):
    df = DataFrame(data, columns=[i+1 for i in range(data.shape[1])])
    scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')

# %%
def graph_plot(graph_object):
    nx.draw(graph_object, nx.spring_layout(graph_object))
    plt.axis('tight')
    plt.show()
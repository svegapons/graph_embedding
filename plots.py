# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 15:39:17 2014

@author: jm
"""

import matplotlib.pyplot as plt


#%%
def heatmap(data):
    """Print a heatmap of the data
    for better understanding the range
    of values"""
    
    plt.title("Heatmap")
    obj = plt.matshow(data)
    plt.colorbar(obj)
    plt.show()
    
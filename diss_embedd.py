# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 15:38:00 2015

@author: jm
"""

import numpy as np



def diss_embedd(A, B, K):
    """Dissimilarity embedding techiqnue
    Input: 1. Adjacency matrix A and B representing two different graphs
           2. Constant K for penalizing missing edges
    Output: Dissimilarity vector y of A and B"""


    #  sanity check
    width_A, height_A = A.shape
    width_B, height_B = B.shape

    if width_A != width_B and height_A != height_B:
        print("Arrays must be of similar size")
        return 0

    triuA = np.triu(A) - np.eye(width_A, height_A) * np.diag(A)
    triuB = np.triu(B) - np.eye(width_B, height_B) * np.diag(B)
    y = np.zeros([width_A, height_A])

    for i in range(width_A):
        for j in range(i+1, height_A):
            if triuA[i, j] == 0:
                y[i, j] = K
            elif triuB[i, j] == 0:
                y[i, j] = K
            else: y[i, j] = np.abs(triuA[i, j]-triuB[i, j])

    return np.sum(y)

def embedd_list(adj_list, K):
    """List of adjacency matrices to compute the dissimilarity between them
    Input: 1. List of adjacency matrices
           2. Constant K for penalizing missing edges
    Ouput: Dissimilarity matrix Y for all adjancecy inputs"""

    width_lst = len(adj_list)

    Y = np.zeros([width_lst, width_lst])

    for lst_x in range(width_lst):
        for lst_y in range(width_lst):
            if lst_x == lst_y:
                Y[lst_x, lst_y] = 0
            else:
                Y[lst_x, lst_y] = diss_embedd(adj_list[lst_x],  adj_list[lst_y], K)

    return Y




if __name__ == "__main__":
    A = np.array([[1,2,3,4,5], [2,3,6,7,9],
                  [0.8,0.6,10,0,7], [-0.8,-05,-0.9,0,0], [0.8,0.9,0.6,0.4,0]])
    B = np.array([[0.8,0.6,10,0,7], [-0.8,-05,-0.9,0,0],
                  [0.8,0.9,0.6,0.4,0], [2,3,6,7,9], [1,2,3,4,5]])
    C = np.array([[1,2,3,4,5], [2,3,6,7,9],
                  [0.8,0.6,10,0,7], [-0.8,-05,-0.9,0,0], [0.8,0.9,0.6,0.4,0]])
    D = np.array([[0.8,0.6,10,0,7], [-0.8,-05,-0.9,0,0],
                  [0.8,0.9,0.6,0.4,0], [2,3,6,7,9], [1,2,3,4,5]])
#    test = diss_embedd(A, B, 100)
    adj_lst = [A, B, C, D]
    K = 100
    X = embedd_list(adj_lst, K)


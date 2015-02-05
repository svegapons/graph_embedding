# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:28:27 2015

@author: jm
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 17:36:12 2015

@author: jm
"""

import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from misc import percentage_removed

#####Encoding techniques based on complex network measures###########

#####Node based encoding#########
def node_centrality(X):
    """
    based on networkx function: degree_centrality
    """
#    graph_position = np.arange(0, 192, 48)
#    pos_counter = 0
    XX = np.zeros((X.shape[0], np.sqrt(X.shape[1])))
    for i, value in enumerate(X):
        adj_mat = value.reshape((np.sqrt(len(value)), -1)) # reshape to 254x254
        adj_mat = (adj_mat - np.min(adj_mat)) / (np.max(adj_mat) - np.min(adj_mat))
        adj_mat = 1 - adj_mat

        thres = None
        percent, th, adj_mat, triu = percentage_removed(adj_mat, 0.9945, thres, "node_centrality") #0.98945
        print("percent = {0}, threshold position = {1}, threshold = {2}\n".format(percent, th, triu[th]))
#        th = np.mean(adj_mat)
#        th = np.mean(adj_mat) + 0.22 #22
#        adj_mat = np.where(adj_mat > th, 1., 0.) # binary adjacency matrix

        g = nx.from_numpy_matrix(adj_mat)
        print "Graph Nodes = {0}, Graph Edges = {1} ".format(g.number_of_nodes(), g.number_of_edges())
        print "Edge kept ratio, {0}".format(float(g.number_of_edges())/((g.number_of_nodes()*(g.number_of_nodes()-1))/2))

#        graph plotting
#        if (pos_counter > 3): pos_counter = 0
#        nfigs = False
#        from_graph = 4
#        to_graph = 8
#        graph_plot(g, i, graph_position[pos_counter], from_graph, to_graph, nfigs, \
#                   subplot = True)
#        pos_counter += 1

        deg_cent = nx.degree_centrality(g)
        node_cent = np.zeros(g.number_of_nodes())

        for k in deg_cent:
            node_cent[k] = deg_cent[k]
        XX[i] = node_cent
        print "graph {0} => mean {1}, min {2}, max {3}\n".format(i, np.mean(XX[i]), np.min(XX[i]), np.max(XX[i]))

    return XX



def node_closeness_centrality(X):
    """
    based on networkx function: closeness_centrality
    """
    XX = np.zeros((X.shape[0], np.sqrt(X.shape[1])))
    for i, value in enumerate(X):
        adj_mat = value.reshape((np.sqrt(len(value)), -1))
        adj_mat = (adj_mat - np.min(adj_mat)) / (np.max(adj_mat) - np.min(adj_mat))
        adj_mat = 1 - adj_mat

#        th = np.mean(adj_mat) - 0.23
#        adj_mat = np.where(adj_mat < th, adj_mat, 0.)

        percent, th, adj_mat, triu = percentage_removed(adj_mat, 0.27) # in this context the percentage
        print("percent = {0}, threshold position = {1}, threshold = {2}\n".format(percent, th, triu[th]))

        g = nx.from_numpy_matrix(adj_mat)
        print "Graph Nodes = {0}, Graph Edges = {1} ".format(g.number_of_nodes(), g.number_of_edges())
        print "\nEdge kept ratio, {0}".format(float(g.number_of_edges())/((g.number_of_nodes()*(g.number_of_nodes()-1))/2))

        deg_cent = nx.closeness_centrality(g, normalized=True)
        node_cent = np.zeros(g.number_of_nodes())

        for k in deg_cent:
            node_cent[k] = deg_cent[k]
        XX[i] = node_cent
        print "graph {0} => mean {1}, min {2}, max {3}".format(i, np.mean(XX[i]), np.min(XX[i]), np.max(XX[i]))

    return XX



def node_betweeness_centrality(X):
    """
    based on networkx function: betweenness_centrality
    """
    XX = np.zeros((X.shape[0], np.sqrt(X.shape[1])))
    for i, value in enumerate(X):
        adj_mat = value.reshape((np.sqrt(len(value)), -1))
        adj_mat = (adj_mat - np.min(adj_mat)) / (np.max(adj_mat) - np.min(adj_mat))
        adj_mat = 1 - adj_mat

#        th = np.mean(adj_mat) - 0.1
#        adj_mat = np.where(adj_mat < th, adj_mat, 0.)

        percent, th, adj_mat, triu = percentage_removed(adj_mat, 0.6) #34
        print("percent = {0}, threshold position = {1}, threshold = {2}\n".format(percent, th, triu[th]))

        g = nx.from_numpy_matrix(adj_mat)
        print "Graph Nodes = {0}, Graph Edges = {1} ".format(g.number_of_nodes(), g.number_of_edges())
        print "\nEdge kept ratio, {0}".format(float(g.number_of_edges())/((g.number_of_nodes()*(g.number_of_nodes()-1))/2))

        deg_cent = nx.betweenness_centrality(g)
#        deg_cent = nx.betweenness_centrality(g)*1000

        node_cent = np.zeros(g.number_of_nodes())

        for k in deg_cent:
            node_cent[k] = deg_cent[k]
#        node_cent = deg_cent.values()
        XX[i] = node_cent
        print "graph {0} => mean {1}, min {2}, max {3}".format(i, np.mean(XX[i]), np.min(XX[i]), np.max(XX[i]))
#    XX = XX*100
    XX = StandardScaler().fit_transform(XX.T).T

    return XX


def node_eigenvector_centrality(X):
    """
    based on networkx function: eigenvector_centrality
    """
    XX = np.zeros((X.shape[0], np.sqrt(X.shape[1])))
    for i, value in enumerate(X):
        adj_mat = value.reshape((np.sqrt(len(value)),-1))
        adj_mat = (adj_mat - np.min(adj_mat)) / (np.max(adj_mat) - np.min(adj_mat))
        adj_mat = 1 - adj_mat

#        th = np.mean(adj_mat) - 0.2
#        adj_mat = np.where(adj_mat < th, adj_mat, 0.)

        percent, th, adj_mat, triu = percentage_removed(adj_mat, 0.78)
        print("percent = {0}, threshold position = {1}, threshold = {2}\n".format(percent, th, triu[th]))

        g = nx.from_numpy_matrix(adj_mat)
        print "Graph Nodes = {0}, Graph Edges = {1} ".format(g.number_of_nodes(), g.number_of_edges())
        print "\nEdge kept ratio, {0}".format(float(g.number_of_edges())/((g.number_of_nodes()*(g.number_of_nodes()-1))/2))

        deg_cent = nx.eigenvector_centrality(g, max_iter=10000)
        node_cent = np.zeros(g.number_of_nodes())

        for k in deg_cent:
            node_cent[k] = deg_cent[k]
        XX[i] = node_cent
        print "graph {0} => mean {1}, min {2}, max {3}".format(i, np.mean(XX[i]), np.min(XX[i]), np.max(XX[i]))
#    XX = XX*100
    ss = StandardScaler()
    XX = ss.fit_transform(XX.T).T

    return XX


def node_communicability_centrality(X):
    """
    based on networkx function: communicability_centrality
    """
    XX = np.zeros((X.shape[0], np.sqrt(X.shape[1])))
    for i, value in enumerate(X):
        adj_mat = value.reshape((np.sqrt(len(value)),-1))
        adj_mat = (adj_mat - np.min(adj_mat)) / (np.max(adj_mat) - np.min(adj_mat))
        adj_mat = 1 - adj_mat

#        th = np.mean(adj_mat) - 0.1
#        adj_mat = np.where(adj_mat < th, adj_mat, 0.)
        print("\n========== Node communicability centrality ==========\n")
        percent, th, adj_mat, triu = percentage_removed(adj_mat, 0.76) #96
        print("percent = {0}, threshold position = {1}, threshold = {2}\n".format(percent, th, triu[th]))

        g = nx.from_numpy_matrix(adj_mat)
        print "Graph Nodes = {0}, Graph Edges = {1} ".format(g.number_of_nodes(), g.number_of_edges())
        print "\nEdge kept ratio, {0}".format(float(g.number_of_edges())/((g.number_of_nodes()*(g.number_of_nodes()-1))/2))

        deg_cent = nx.communicability_centrality(g)
        node_cent = np.zeros(g.number_of_nodes())

        for k in deg_cent:
            node_cent[k] = deg_cent[k]
        XX[i] = node_cent
        print "graph {0} => mean {1}, min {2}, max {3}".format(i, np.mean(XX[i]), np.min(XX[i]), np.max(XX[i]))
#    XX = XX*100
    ss = StandardScaler()
    XX = ss.fit_transform(XX.T).T

    return XX


def node_load_centrality(X):
    """
    based on networkx function: load_centrality
    """
    XX = np.zeros((X.shape[0], np.sqrt(X.shape[1])))
    for i, value in enumerate(X):
        adj_mat = value.reshape((np.sqrt(len(value)),-1))
        adj_mat = (adj_mat - np.min(adj_mat)) / (np.max(adj_mat) - np.min(adj_mat))
        adj_mat = 1 - adj_mat

#        th = np.mean(adj_mat) - 0.05
#        adj_mat = np.where(adj_mat < th, adj_mat, 0.)
        print("\n========== Node load centrality ==========\n")
        percent, th, adj_mat, triu = percentage_removed(adj_mat, 0.86)
        print("percent = {0}, threshold position = {1}, threshold = {2}\n".format(percent, th, triu[th]))

        g = nx.from_numpy_matrix(adj_mat)
        print "Graph Nodes = {0}, Graph Edges = {1} ".format(g.number_of_nodes(), g.number_of_edges())
        print "\nEdge kept ratio, {0}".format(float(g.number_of_edges())/((g.number_of_nodes()*(g.number_of_nodes()-1))/2))

        deg_cent = nx.load_centrality(g, weight = 'weight')
        node_cent = np.zeros(g.number_of_nodes())

        for k in deg_cent:
            node_cent[k] = deg_cent[k]
        XX[i] = node_cent
        print "graph {0} => mean {1}, min {2}, max {3}".format(i, np.mean(XX[i]), np.min(XX[i]), np.max(XX[i]))
#    XX = XX*100
    ss = StandardScaler()
    XX = ss.fit_transform(XX.T).T

    return XX


def node_current_flow_closeness_centrality(X):
    """
    based on networkx function: current_flow_closeness_centrality
    """
    XX = np.zeros((X.shape[0], np.sqrt(X.shape[1])))
    for i, value in enumerate(X):
        adj_mat = value.reshape((np.sqrt(len(value)),-1))
        adj_mat = (adj_mat - np.min(adj_mat)) / (np.max(adj_mat) - np.min(adj_mat))
        adj_mat = 1 - adj_mat

#        th = np.mean(adj_mat) - 0.05
#        adj_mat = np.where(adj_mat < th, adj_mat, 0.)
        print("\n========== Node current flow closeness centrality ==========\n")
        percent, th, adj_mat, triu = percentage_removed(adj_mat, 0.64) #74
        print("percent = {0}, threshold position = {1}, threshold = {2}\n".format(percent, th, triu[th]))

        g = nx.from_numpy_matrix(adj_mat)
        print "Graph Nodes = {0}, Graph Edges = {1} ".format(g.number_of_nodes(), g.number_of_edges())
        print "\nEdge kept ratio, {0}".format(float(g.number_of_edges())/((g.number_of_nodes()*(g.number_of_nodes()-1))/2))

        deg_cent = nx.current_flow_closeness_centrality(g)
        node_cent = np.zeros(g.number_of_nodes())

        for k in deg_cent:
            node_cent[k] = deg_cent[k]
        XX[i] = node_cent
        print "graph {0} => mean {1}, min {2}, max {3}".format(i, np.mean(XX[i]), np.min(XX[i]), np.max(XX[i]))
#    XX = XX*100
    ss = StandardScaler()
    XX = ss.fit_transform(XX.T).T

    return XX

########Edge based encoding########################
def edge_betweeness_centrality(X):
    """
    based on networkx function: edge_betweenness_centrality
    """
    XX = np.zeros(X.shape)
    for i, value in enumerate(X):
        adj_mat = value.reshape((np.sqrt(len(value)),-1))
        adj_mat = (adj_mat - np.min(adj_mat)) / (np.max(adj_mat) - np.min(adj_mat))
        adj_mat = 1 - adj_mat

#        th = np.mean(adj_mat) + 0.1
#        adj_mat = np.where(adj_mat < th, adj_mat, 0.)

#        percent, th, adj_mat, triu = percentage_removed(adj_mat, 0.43) # 43 #63 #73
#        print("percent = {0}, threshold position = {1}, threshold = {2}\n".format(percent, th, triu[th]))

        g = nx.from_numpy_matrix(adj_mat)
        print "Graph Nodes = {0}, Graph Edges = {1} ".format(g.number_of_nodes(), g.number_of_edges())
        print "\nEdge kept ratio, {0}".format(float(g.number_of_edges())/((g.number_of_nodes()*(g.number_of_nodes()-1))/2))

        bet_cent = nx.edge_betweenness_centrality(g, weight = 'weight', normalized = True)
        edge_cent = np.zeros(adj_mat.shape)

        for k in bet_cent:
            edge_cent[k[0],k[1]] = bet_cent[k]
        XX[i] = edge_cent.reshape(-1)
        print "graph {0} => mean {1}, min {2}, max {3}".format(i, np.mean(XX[i]), np.min(XX[i]), np.max(XX[i]))

    return XX
# -*- coding: utf-8 -*-
"""
Comparison of different embedding techniques for brain decoding
"""
import numpy as np
import networkx as nx
import os
# import pdb
import pandas as pd
from sklearn.cross_validation import cross_val_score, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
#from sklearn.lda import LDA
#from sklearn.qda import QDA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import ElasticNet, SGDRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plots import *
from misc import *
from gk_weisfeiler_lehman import GK_WL


def process_data(path_folder, n_files_per_subj, desc_file, graph_metric):
    """
    Load the dataset.
    Parameters:
    ----------
    path_folder: string
        Path to the folder containing all data
    n_files_per_subj: int
        Number of files associated to each subject
    desc_file: string
        Path to the file with the description of the experiment

    Example:
    -------
    These are the four files associated to subject KKI_1018959 in dataset ADHD:
    'KKI_1018959_connectivity_matrix_file.txt'
    'KKI_1018959_region_names_abbrev_file.txt'
    'KKI_1018959_region_names_full_file.txt'
    'KKI_1018959_region_xyz_centers_file.txt'
    """

    #Reading the description file and extracting the name of the samples and
    # the associated class
    desc = pd.read_csv(desc_file, sep='\t')
    name = desc['upload_data.network_name']
    pool = desc['upload_data.subject_pool']

    dt_name = dict()
    for i, value in enumerate(name):
        dt_name[value] = i

    dt_cls  = dict()
    cs = np.unique(pool)
    for i, value in enumerate(cs):
        dt_cls[value] = i


    dirs = os.listdir(path_folder)
    dirs.sort()
    connectivity_mat = []
    reg_abbrev = []
    reg_full = []
    reg_cents = []
    names = []
    y = []

    #Reading the data files
    for i, value in enumerate(dirs):
        spl = value.split('_')
        if spl[-2] == 'matrix':
            connectivity_mat.append(np.loadtxt(os.path.join(path_folder, value)))
            nm = '_'.join(spl[:-3])
            names.append(nm)
            y.append(dt_cls[pool[dt_name[nm]]])
            assert cs[y[-1]] == pool[dt_name[nm]], 'wrong class assignment'
        elif spl[-2] == 'centers':
            reg_cents.append(np.fromfile(os.path.join(path_folder, value)).reshape((-1, 3)))


    X = np.zeros((len(connectivity_mat), connectivity_mat[0].shape[0] * connectivity_mat[0].shape[1]))

    for i, value in enumerate(connectivity_mat):
        X[i] = value.reshape(-1)

    y = np.array(y)
    X = np.where(X == np.inf, 0, X)

#==============================================
#    Settigs for simple embedding
#==============================================
#    threshold = 0.15
#    threshold = None
#    X = variance_threshold(X, y, threshold)


#============================================
#   Settings for graph kernels
#============================================
    graphs = [0]*len(X)
    for (x, y), val in np.ndenumerate(X):
        data = val.reshape(264, 264)
        data = np.where(data > np.mean(data), 1., 0.) # binarize
        data = data - np.diag(np.diag(data))
        graph = nx.from_numpy_matrix(data)
        graphs[x] = graph

#    idxs = (y==0) + (y==1)
#    X = X[idxs]
#    y = y[idxs]
#    y[y==3] = 1

#
#    np.save('E:\Python\Data\UCLA_Autism\X', X)
#    np.save('E:\Python\Data\UCLA_Autism\y', y)

#    X = np.load('E:\Python\Data\UCLA_Autism\X.npy')
#    X = np.load('E:\Python\Data\UCLA_Autism\X_node_cent.npy1')
#    y = np.load('E:\Python\Data\UCLA_Autism\y.npy')

    print "\n=====Class distribution=====\n"
    print "Autism Spectrum Disorder: %s, Typically Developing: %s " %(sum(y == 0), sum(y == 1))
    print "Dimenesioality of training data => X: {0}".format(np.shape(X))
    print "Dimensionality of labels => y: {0} \n".format(np.shape(y))


#    if (graph_metric == "1"):    # node_centrality
#        print "=====Edge Information Node Centrality====="
#        X = node_centrality(X)
#    elif (graph_metric == "2"):    # node_centrality
#        print "=====Edge Information Node Closeness Centrality====="
#        X = node_closeness_centrality(X)
#    elif (graph_metric == "3"):  # node_betweeness
#        print "=====Edge Information Node Betweeness Centrality====="
#        X = node_betweeness_centrality(X)
#    elif (graph_metric == "4"):  # edge_betweeness
#        print "=====Edge Information Edge Betweeness Centrality====="
#        X = edge_betweeness_centrality(X)
#    elif (graph_metric == "5"):   # node_eigenvector
#        print "=====Edge Information Node Eigenvector Centrality====="
#        X = node_eigenvector_centrality(X)
#    elif (graph_metric == "6"):  # node_communicability
#        print "=====Edge Information Node Communicability Centrality====="
#        X = node_communicability_centrality(X)
#    elif (graph_metric == "7"):  # node_load_centrality
#        print "=====Edge Information Node Load Centrality====="
#        X = node_load_centrality(X)
#    elif (graph_metric == "8"):  # node_current_flow_centrality
#        print "=====Edge Information Node Current Flow Closeness Centrality====="
#        X = node_current_flow_closeness_centrality(X)
#    else:
#        print "Wrong Choice!"
#        return

#    X = np.hstack((node_centrality(X), node_closeness_centrality(X)))

#    X = X[:,range(0,200)]
#    np.save('E:\Python\Data\UCLA_Autism\X_node_cent', X)
#    X = node_closeness_centrality(X)
#    X = node_betweeness_centrality(X)
#    X = edge_betweeness_centrality(X)

#    ####GRID SEARCH
#    param_grid = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']},\
#                     {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
#
#    clf = GridSearchCV(SVC(), param_grid, scoring = 'roc_auc', cv=5, refit = True, verbose=2)
#    ####
#
     ###NORMALIZATION
#    ss = StandardScaler()
#    X = ss.fit_transform(X.T).T

# ============= Plotting =============

#    plt.title('Histogram')
#    for i, value in enumerate(X):
#        plt.plot(range(X.shape[1]), value, 'b-o' if y[i] == 0 else 'm-o')
#    plt.show()

#    heatmap(connectivity_mat)
#    scatter_mat(X[:, :10])

# ============= Pre-processing =============

#   it works with embedding techniques 3,5,6
#    threshold = 0.15
#    threshold = None
#    X = variance_threshold(X, y, threshold)

#    svc = SVC(kernel="linear", C=1)
#    rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
#    rfe.fit(X, y)

#    Tree-based feautre selection
#     clf = ExtraTreesClassifier()
#     X_new = clf.fit(X, y).transform(X)

# ============= Dimensionality reduction/classification =============

#   LDA vs. QDA classifiers
#    for i, (X, y) in enumerate([dataset_fixed_cov(), dataset_cov()]):
#        # LDA
#        lda = LDA()
#        y_pred = lda.fit(X, y, store_covariance=True).predict(X)
#        splot = plot_data(lda, X, y, y_pred, fig_index=2 * i + 1)
#        plot_lda_cov(lda, splot)
#        plt.axis('tight')
#
#        # QDA
#        qda = QDA()
#        y_pred = qda.fit(X, y, store_covariances=True).predict(X)
#        splot = plot_data(qda, X, y, y_pred, fig_index=2 * i + 2)
#        plot_qda_cov(qda, splot)
#        plt.axis('tight')
#    plt.suptitle('LDA vs QDA')
#    plt.show()


#    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
#    X_kpca = kpca.fit_transform(X)
#    X_back = kpca.inverse_transform(X_kpca)
#    pca = PCA()
#    X_pca = pca.fit_transform(X)

# ============= Pipeline feature selection/classification =============

#    In this snippet we make use of a sklearn.svm.LinearSVC to evaluate feature
#    importances and select the most relevant features. Then,
#   a sklearn.ensemble.RandomForestClassifier is trained on the
#    transformed output, i.e. using only relevant features.
#    You can perform similar operations with the other feature
#   selection methods and also classifiers that provide a way to
#   evaluate feature importances of course

#    clf = Pipeline([
#      ('feature_selection', LinearSVC(penalty="l1")),
#      ('classification', RandomForestClassifier())
#    ])
#    clf.fit(X, y)

#   clf = make_pipeline(Binarizer(), MultinomialNB())
#   clf.fit(X, y)

# ============= Feature importances with forests of trees =============

# use of forests of trees to evaluate the importance of features on a
# classification task

#    # Build a forest and compute the feature importances
#    forest = ExtraTreesClassifier(n_estimators=250,
#                                  random_state=0)
#
#    forest.fit(X, y)
#    importances = forest.feature_importances_
#    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#                 axis=0)
#    indices = np.argsort(importances)[::-1]
#
#    # Print the feature ranking
#    print("Feature ranking:")
#
#    for f in range(10):
#        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
#    # Plot the feature importances of the forest
#    plt.figure()
#    plt.title("Feature importances")
#    plt.bar(range(10), importances[indices],
#           color="r", yerr=std[indices], align="center")
#    plt.xticks(range(10), indices)
#    plt.xlim([-1, 10])
#    plt.show()

#    clf_tmp = ExtraTreesClassifier()
#    X = clf_tmp.fit(X, y).transform(X)
#

# ============= Gaussian Naive Bayes classifier =============

#    from sklearn.naive_bayes import GaussianNB
#    gnb = GaussianNB()
#    y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
#    print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))

    score = []
    n_folds = 10
    kfold = StratifiedKFold(y, n_folds)
    for train_ind, test_ind in kfold:
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
#
#        clf = Pipeline([
#                      ('feature_selection', LinearSVC(penalty="l1", dual=False)),
#                      ('classification', RandomForestClassifier())
#                      ])
#        clf.fit(X_train, y_train)

        clf = LinearSVC(C=10000, loss='l2', random_state = 1231)
##        clf = ElasticNet(alpha=0.0001, l1_ratio=0.15)
##        clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=300, max_depth=13)
        clf.fit(X_train, y_train)

#        clf = pca_svm(X_train, y_train)
#        clf = multinomial_bayes(X_train, y_train)
#        clf = anova_svm(X_train, y_train)
        pred = clf.predict(X_test)
        score.append(accuracy_score(y_test, pred))
#
    print "Score: %s" %(score)
    print 'Mean AUC: %s, std: %s' %(np.mean(score), np.std(score))



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
#        adj_mat = 1 - adj_mat

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

        percent, th, adj_mat, triu = percentage_removed(adj_mat, 0.34) #34
        print("percent = {0}, threshold position = {1}, threshold = {2}\n".format(percent, th, triu[th]))

        g = nx.from_numpy_matrix(adj_mat)
        print "Graph Nodes = {0}, Graph Edges = {1} ".format(g.number_of_nodes(), g.number_of_edges())
        print "\nEdge kept ratio, {0}".format(float(g.number_of_edges())/((g.number_of_nodes()*(g.number_of_nodes()-1))/2))

        deg_cent = nx.betweenness_centrality(g, k = 50, normalized = True)
        node_cent = np.zeros(g.number_of_nodes())

        for k in deg_cent:
            node_cent[k] = deg_cent[k]
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

        percent, th, adj_mat, triu = percentage_removed(adj_mat, 0.43) # 43 #63 #73
        print("percent = {0}, threshold position = {1}, threshold = {2}\n".format(percent, th, triu[th]))

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




## main ##
if __name__=='__main__':

    # Path to the folder containing the connectivity matrices
    path_folder='UCLA_Autism/UCLA_Autism_fMRI'
    # Path to the file containing the description of the experiments.
    # This file is used to extract the clas associated to each sample
    desc_file='UCLA_Autism/UCLA_Autism_fMRI.csv'
    # Number of files associated to each subject
    n_files_per_subj = 4
    # Choose one of the following graph metrics
    print "Please choose [1-8] of the following graph metrics:\n \
    1. node_centrality \n \
    2. node_closeness_centrality \n \
    3. node_betweeness_centrality \n \
    4. edge_betweeness_centrality \n \
    5. node_eigenvector_centrality\n \
    6. node_communicability_centrality \n \
    7. node_load_centrality \n \
    8. node_current_flow_centrality \n"
    graph_metric = raw_input()
    process_data(path_folder, n_files_per_subj, desc_file, graph_metric)

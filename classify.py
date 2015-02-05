# -*- coding: utf-8 -*-
"""
Comparison of different embedding techniques for brain decoding
"""
import numpy as np
# import pdb
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
from load import process_data
#from plots import *
#from misc import *
#from gk_weisfeiler_lehman import GK_WL
#from gk_shortest_path import GK_SP


#==============================================
#    Settigs for simple embedding
#==============================================
#    threshold = 0.15
#    threshold = None
#    X = variance_threshold(X, y, threshold)


#%%
#============================================
#   Settings for graph kernels
#============================================
#    graphs = [0]*len(X)
#    counter = 0
#    for val in X:
#        data = val.reshape(264, 264)
#        data = np.where(data > np.mean(data), 1., 0.) # binarize
#        data = data - np.diag(np.diag(data))
#        graph = nx.from_numpy_matrix(data)
#        graphs[counter] = graph
#        counter += 1

#%%
#============================================
#   GK_WL Kernels
#============================================
#    gk_wl_obj = GK_WL()
#    kernel = gk_wl_obj.compare_list_normalized(graphs, 1, False)

#%%
#============================================
#   GK_SP Kernels
#============================================
#    gk_sp_obj = GK_SP()
#    kernel = gk_sp_obj.compare_list_normalized(graphs)


#    clf = SVC(C = 10000, kernel = 'precomputed', random_state = 1231)
#    clf = SVC(C=1)
#    perform grid search
#    parameters = {'kernel':('linear', 'rbf', 'poly', 'rbf', 'sigmoid', 'precomputed'), 'C':[1, 10]}
#    parameters = {'kernel':['precomputed'], 'C':[1, 10, 100, 1000, 10000], \
#                  'tol':[1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'random_state':[1231]}
#    gsearch = GridSearchCV(clf, parameters)
#    accuracies = gsearch.score(kernel, y)
#    accuracies = cross_val_score(gsearch, kernel, y)
#    print "Score: %s" %(accuracies)
#    print 'Mean AUC: %s, std: %s' %(np.mean(accuracies), np.std(accuracies))


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

#%%
# ============= Gaussian Naive Bayes classifier =============

#    from sklearn.naive_bayes import GaussianNB
#    gnb = GaussianNB()
#    y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
#    print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))


def svm(X, y):

    Cs = list(10.0 ** np.arange(-1, 6))
#    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    degrees = range(3, 10)
    gammas = list(2.0 ** np.arange(-4, 3))
    coef0s = list(np.logspace(-2, 2, 5))
    param_grid = dict(C = Cs, degree = degrees, gamma = gammas, coef0 = coef0s, random_state = [1231])

    score = []
    kfold = StratifiedKFold(y, n_folds = 10)
    for train_ind, test_ind in kfold:
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
#
#        clf = Pipeline([
#                      ('feature_selection', LinearSVC(penalty="l1", dual=False)),
#                      ('classification', RandomForestClassifier())
#                      ])
#        clf.fit(X_train, y_train)

#        clf = LinearSVC(C=10000, loss='l2', random_state = 1231)
##        clf = ElasticNet(alpha=0.0001, l1_ratio=0.15)
##        clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=300, max_depth=13)

#        X_train, var_obj1, var_obj2 = variance_threshold(X_train, y_train, 0.15)
#        X_test = var_obj1.transform(X_test)
#        X_test = var_obj2.transform(X_test)

#        gs = pca_logisticRegression(X_train, y_train)

        clf = SVC()
        gs = GridSearchCV(clf, param_grid, scoring='roc_auc', cv=StratifiedKFold(y_train, n_folds=10), refit=True, verbose=1)
        gs.fit(X_train, y_train)
        pred = gs.predict(X_test) # problem X.shape[1] = 264 but it should be 20 like

#        print("Best score: {0}, parameters: {1}, classifier: {2}".format(gs.best_score_, gs.best_params_, gs.best_estimator_))
#        rnd_feat = np.random.choice(20, 20, replace=False)
#        pred = clf.predict(X_test)
        # X_train.shape[1] = 20

#        clf = pca_svm(X_train, y_train)
#        clf = multinomial_bayes(X_train, y_train)
#        clf = anova_svm(X_train, y_train)
#        pred = clf.predict(X_test)
        score.append(accuracy_score(y_test, pred))
#
    print "Score: %s" %(score)
    print 'Mean AUC: %s, std: %s' %(np.mean(score), np.std(score))



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

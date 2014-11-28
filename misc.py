# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 17:58:09 2014

@author: jm
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion, make_union
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer
from sklearn.feature_selection import SelectKBest, f_regression, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

#The estimators of a pipeline are stored as a list in the steps attribute:
#>>>
#
#>>> clf.steps[0]
#('reduce_dim', PCA(copy=True, n_components=None, whiten=False))
#
#and as a dict in named_steps:
#>>>
#
#>>> clf.named_steps['reduce_dim']
#PCA(copy=True, n_components=None, whiten=False)
#
#Parameters of the estimators in the pipeline can be accessed using the <estimator>__<parameter> syntax:
#>>>
#
#>>> clf.set_params(svm__C=10)
#Pipeline(steps=[('reduce_dim', PCA(copy=True, n_components=None,
#    whiten=False)), ('svm', SVC(C=10, cache_size=200, class_weight=None,
#    coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=-1,
#    probability=False, random_state=None, shrinking=True, tol=0.001,
#    verbose=False))])
#
#This is particularly important for doing grid searches:
#>>>
#
#>>> from sklearn.grid_search import GridSearchCV
#>>> params = dict(reduce_dim__n_components=[2, 5, 10],
#...               svm__C=[0.1, 10, 100])
#>>> grid_search = GridSearchCV(clf, param_grid=params)


# Pipeline: chaining estimators

def pca_svm(train_data, labels):
    estimators = [('reduce_dim', PCA()), ('svm', SVC())]
    clf = Pipeline(estimators)
    clf.fit(train_data, labels)

def multinomial_bayes(train_data, labels):
    clf = make_pipeline(Binarizer(), MultinomialNB())
    clf.fit(train_data, labels)

# FeatureUnion: Combining feature extractors

def pca_kpca(train_data, labels):
    estimators = make_union(PCA(), TruncatedSVD(), KernelPCA())
#    estimators = [('linear_pca', PCA()), ('kernel_pca', KernelPCA())]
    combined = FeatureUnion(estimators)
    combined.fit(train_data, labels) # combined.fit_tranform(tain_data, labels)

# Pipelining: chaining anova with svm
def anova_svm(train_data, labels):
    # ANOVA SVM-C
    # 1) anova filter, take 3 best ranked features
    anova_filter = SelectKBest(f_regression, k = 3)
    # 2) svm
    clf = SVC(kernel = 'linear')

    anova_svm = make_pipeline(anova_filter, clf)
    anova_svm.fit(train_data, labels)
    anova_svm.predict(train_data)

# Pipelining: chaining a PCA and a logistic regression
def pca_logisticRegression(train_data, labels):
    logistic = LogisticRegression()
    pca = PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    ###############################################################################
    # Plot the PCA spectrum
    pca.fit(train_data)

    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')

    ###############################################################################
    # Prediction

    n_components = [20, 40, 64]
    Cs = np.logspace(-4, 4, 3)

    #Parameters of pipelines can be set using ‘__’ separated parameter names:

    estimator = GridSearchCV(pipe,
                             dict(pca__n_components=n_components,
                                  logistic__C=Cs))
    estimator.fit(train_data, labels)

    plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    plt.legend(prop=dict(size=12))
    plt.show()

# SVM-Anova: SVM with univariate feature selection
# how to perform univariate feature before running a SVC (support vector
# classifier) to improve the classification scores
def svm_anova(train_data, labels):
    ###############################################################################
    # Create a feature-selection transform and an instance of SVM that we
    # combine together to have an full-blown estimator

    transform = SelectPercentile(f_classif)

    clf = Pipeline([('anova', transform), ('svc', svm.SVC(C=1.0))])

    ###############################################################################
    # Plot the cross-validation score as a function of percentile of features
    score_means = list()
    score_stds = list()
    percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

    for percentile in percentiles:
        clf.set_params(anova__percentile = percentile)
        # Compute cross-validation score using all CPUs
        this_scores = cross_val_score(clf, train_data, labels, n_jobs=1)
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())

    plt.errorbar(percentiles, score_means, np.array(score_stds))

    plt.title(
        'Performance of the SVM-Anova varying the percentile of features selected')
    plt.xlabel('Percentile')
    plt.ylabel('Prediction rate')

    plt.axis('tight')
    plt.show()

# Concatenating multiple feature extraction methods
def concat_feature_extractors(train_data, labels):
    # This dataset is way to high-dimensional. Better do PCA:
    pca = PCA(n_components = 2)

    # Maybe some original features where good, too?
    selection = SelectKBest(k = 1)

    # Build estimator from PCA and Univariate selection:

    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

    # Use combined features to transform dataset:
    X_features = combined_features.fit(train_data, labels).transform(train_data)

    # Classify:
    svm = SVC(kernel="linear")
    svm.fit(X_features, labels)

    # Do grid search over k, n_components and C:

    pipeline = Pipeline([("features", combined_features), ("svm", svm)])

    param_grid = dict(features__pca__n_components=[1, 2, 3],
                      features__univ_select__k=[1, 2],
                      svm__C=[0.1, 1, 10])

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
    grid_search.fit(train_data, labels)
    print(grid_search.best_estimator_)
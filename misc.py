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
from sklearn.feature_selection import SelectKBest, f_regression, SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import classification_report
import logging
from pprint import pprint
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

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

    clf = Pipeline([('anova', transform), ('svc', SVC(C=1.0))])

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

# Parameter estimation using grid search with cross-validation
def grid_search(train_data, labels):
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels, \
    test_size = 0.5, random_state = 0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C = 1), tuned_parameters, cv = 5, scoring = score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    # Note the problem is too easy: the hyperparameter plateau is too flat and the
    # output model is the same for precision and recall with ties in quality.

# Sample pipeline for text feature extraction and evaluation
def pipeline_feature_extraction(train_data, labels):
    print(__doc__)

    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')


    ###############################################################################
    # Load some categories from the training set
#    categories = [
#        'alt.atheism',
#        'talk.religion.misc',
#    ]
    # Uncomment the following to do the analysis on all the categories
    #categories = None

#    print("Loading 20 newsgroups dataset for categories:")
#    print(categories)

#    data = fetch_20newsgroups(subset='train', categories=categories)
#    print("%d documents" % len(data.filenames))
#    print("%d categories" % len(data.target_names))
#    print()

    ###############################################################################
    # define a pipeline combining a text feature extractor with a simple
    # classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        #'clf__n_iter': (10, 50, 80),
    }

#    if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(train_data, labels)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Randomized search and grid search for hyperparameter estimation
def randomized_search_and_grid_search_for_hyperparameter_estimation(train_data, labels):
    # build a classifier
    clf = RandomForestClassifier(n_estimators = 20)


    # Utility function to report best scores
    def report(grid_scores, n_top = 3):
        top_scores = sorted(grid_scores, key = itemgetter(1), reverse = True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  score.mean_validation_score,
                  np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")


    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(1, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)

    start = time()
    random_search.fit(train_data, labels)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.grid_scores_)

    # use a full grid over all parameters
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [1, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time()
    grid_search.fit(train_data, labels)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.grid_scores_)))
    report(grid_search.grid_scores_)

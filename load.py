# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:27:30 2015

@author: jm
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 17:28:41 2015

@author: jm
"""

import numpy as np
import pandas as pd
import os

def process_data(path_folder, n_files_per_subj, desc_file):
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
#    reg_abbrev = []
#    reg_full = []
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

    # On screen information about data class distribution
    print "\n=====Class distribution=====\n"
    print "Autism Spectrum Disorder: %s, Typically Developing: %s " %(sum(y == 0), sum(y == 1))
    print "Dimenesioality of training data => X: {0}".format(np.shape(X))
    print "Dimensionality of labels => y: {0} \n".format(np.shape(y))

    return X
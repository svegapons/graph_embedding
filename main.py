# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 17:53:09 2015

@author: jm
"""

from load import process_data
#from embed import *
#from classify import *


## main ##
if __name__=='__main__':

    # Path to the folder containing the connectivity matrices
    path_folder='UCLA_Autism/UCLA_Autism_fMRI'
    # Path to the file containing the description of the experiments.
    # This file is used to extract the clas associated to each sample
    desc_file='UCLA_Autism/UCLA_Autism_fMRI.csv'
    # Number of files associated to each subject
    n_files_per_subj = 4
    X = process_data(path_folder, n_files_per_subj, desc_file)
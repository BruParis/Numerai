from sklearn.model_selection import train_test_split
import numpy as np
import itertools
import pandas as pd
import os
import errno
import json

from ..strat import StratConstitution
from ..reader import ReaderCSV, load_h5_eras
from ..common import *

CL_ERAS_CORR_FP = 'eras_corr.csv'
CL_TR_DATA_FP = 'training_data.csv'
CL_TEST_DATA_FP = 'test_data.csv'


# move to data_analysis ?
def correlation_matrix(eras_df):
    eras_input_df = eras_df.drop([TARGET_LABEL], axis=1)
    corr_matrix = eras_input_df.corr()
    corr_matrix.index.names = ['corr_features']

    return corr_matrix


def generate_data_cluster(dirname, cl_data_df, cl_name, cl_fts):
    print("generate data for cluster: ", cl_name)
    cl_dirpath = dirname + '/' + cl_name + '/'

    try:
        os.makedirs(cl_dirpath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Error with : make dir ", cl_dirpath)
            exit(1)

    columns = ['era'] + cl_fts + [TARGET_LABEL]
    cl_ft_data_df = cl_data_df.loc[:, columns]

    eras_corr_matrix = correlation_matrix(cl_ft_data_df)
    # train_df, test_df = train_test_split(cl_ft_data_df, test_size=TEST_RATIO)

    #cl_ft_data_df.to_csv(cl_dirpath + CL_NUMERAI_TR_DATA_FP)
    eras_corr_matrix.to_csv(cl_dirpath + CL_ERAS_CORR_FP)
    # remove this file
    cl_ft_data_df.to_csv(cl_dirpath + CL_TR_DATA_FP)
    # test_df.to_csv(cl_dirpath + CL_TEST_DATA_FP)


def split_data_clusters(dirname):

    strat_c = StratConstitution(dirname + '/' + STRAT_CONSTITUTION_FILENAME)
    strat_c.load()

    for cl_name, cl_c in strat_c.clusters.items():
        cl_eras = cl_c['eras_name']
        cluster_data = load_h5_eras(TRAINING_STORE_H5_FP,
                                    cl_eras).set_index('id')

        cl_fts = cl_c['selected_features']
        generate_data_cluster(dirname, cluster_data, cl_name, cl_fts)

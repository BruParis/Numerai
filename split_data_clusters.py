from sklearn.model_selection import train_test_split
import numpy as np
import itertools
import pandas as pd
import os
import errno
import json

from model_constitution import ModelConstitution
from feature_order import ft_cat_order_idx
from reader_csv import ReaderCSV
from pool_map import pool_map

TARGET_LABEL = 'target_kazutsugi'
TARGET_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
TARGET_FACT_NUMERIC = 4

FST_LAYER_TRAIN_RATIO = 0.20
TEST_RATIO = 0.20

ERA_CL_DIRNAME = "data_clusters"
MODEL_CONSTITUTION_FP = ERA_CL_DIRNAME + '/model_constitution.json'
DATA_LAYER_FP = ERA_CL_DIRNAME + '/numerai_training_data_layer.csv'
CL_NUMERAI_TR_DATA_FP = 'numerai_training_data.csv'
CL_ERAS_CORR_FP = 'eras_corr.csv'
CL_TR_DATA_FP = 'training_data.csv'
CL_TEST_DATA_FP = 'test_data.csv'


def load_data(data_filename):

    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv().set_index("id")

    return data_df


def format_data(data_df):
    input_df = data_df.drop([TARGET_LABEL], axis=1)
    target_df = data_df.loc[:, [TARGET_LABEL]]

    # mutiply target value to get target class index
    target_df = TARGET_FACT_NUMERIC * target_df

    return input_df, target_df


def write_file(dirname, cl_name, filename, data):
    cl_dirpath = dirname + '/' + cl_name

    filepath = cl_dirpath + '/' + filename
    print("write file: ", filepath)
    with open(filepath, 'w') as f:
        data.to_csv(f)


def correlation_matrix(eras_df):
    eras_input_df = eras_df.drop([TARGET_LABEL], axis=1)
    corr_matrix = eras_input_df.corr()
    corr_matrix.index.names = ['corr_features']

    return corr_matrix


def split_by_clusters_eras(data_df, cl_eras):
    eras_ft_data_df = data_df.loc[data_df['era'].isin(cl_eras)]

    fst_layer_df, remaining_df = train_test_split(
        eras_ft_data_df, test_size=FST_LAYER_TRAIN_RATIO)

    res = pd.DataFrame(False, index=remaining_df.index, columns=['fst_layer'])

    return fst_layer_df, res


def generate_data_cluster(cl_data_df, cl_name, cl_fts):
    print("generate data for cluster: ", cl_name)
    cl_dirpath = ERA_CL_DIRNAME + '/' + cl_name + '/'

    try:
        os.makedirs(cl_dirpath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Error with : make dir ", cl_dirpath)
            exit(1)

    columns = ['era'] + cl_fts + [TARGET_LABEL]
    cl_ft_data_df = cl_data_df.loc[:, columns]

    eras_corr_matrix = correlation_matrix(cl_ft_data_df)
    train_df, test_df = train_test_split(cl_ft_data_df, test_size=TEST_RATIO)

    cl_ft_data_df.to_csv(cl_dirpath + CL_NUMERAI_TR_DATA_FP)
    eras_corr_matrix.to_csv(cl_dirpath + CL_ERAS_CORR_FP)
    train_df.to_csv(cl_dirpath + CL_TR_DATA_FP)
    test_df.to_csv(cl_dirpath + CL_TEST_DATA_FP)


def main():

    model_c = ModelConstitution(MODEL_CONSTITUTION_FP)
    model_c.load()

    data_df = load_data(model_c.orig_data_file)
    data_remaining_df = pd.DataFrame()

    for cl_name, cl_c in model_c.clusters.items():
        cl_eras = cl_c['eras_name']
        cluster_data, remaining_data = split_by_clusters_eras(data_df, cl_eras)
        data_remaining_df = pd.concat(
            [data_remaining_df, remaining_data], axis=0)

        cl_fts = cl_c['selected_features']
        generate_data_cluster(cluster_data, cl_name, cl_fts)

    data_df = pd.concat([data_df, data_remaining_df], axis=1)
    data_fst_layer_df = data_df['fst_layer'].fillna(True)
    data_fst_layer_df.index.name = 'id'

    data_fst_layer_df.to_csv(DATA_LAYER_FP)


if __name__ == '__main__':
    main()

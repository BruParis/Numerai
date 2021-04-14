from sklearn.model_selection import train_test_split
import numpy as np
import itertools
import pandas as pd
import os
import errno
import json

from .feature_order import ft_cat_order_idx
from ..reader_csv import ReaderCSV
from ..pool_map import pool_map

TARGET_LABEL = 'target_kazutsugi'
TARGET_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
TARGET_FACT_NUMERIC = 4

FST_LAYER_TRAIN_RATIO = 0.20
TEST_RATIO = 0.20


def write_model_config(subset_name, filepath, eras):
    eras_dict = {'eras': eras}
    with open(filepath, 'w') as f:
        json.dump(eras_dict, f)


def load_data_models(filepath):
    print("load data models")
    with open(filepath, 'r') as f:
        data_models = json.load(f)

    # need to re-order after json load (?)
    for k, v in data_models.items():
        if 'data_subset' in k:
            ft_l = v['features']
            v['features'] = sorted(ft_l, key=lambda ft: ft_cat_order_idx(ft))

    return data_models


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


def write_file(dirname, subset_name, filename, data):
    subset_dirname = dirname + '/' + subset_name
    try:
        os.makedirs(subset_dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Error with : make dir ", subset_dirname)
            exit(1)

    filepath = subset_dirname + '/' + filename
    print("write file: ", filepath)
    with open(filepath, 'w') as f:
        data.to_csv(f)


def correlation_matrix(eras_df):
    eras_input_df = eras_df.drop([TARGET_LABEL], axis=1)
    corr_matrix = eras_input_df.corr()
    corr_matrix.index.names = ['corr_features']

    return corr_matrix


def split_fst_layer_data(data_df, eras_ft):
    eras = eras_ft['eras']
    eras_ft_data_df = data_df.loc[data_df['era'].isin(eras)]

    fst_layer_df, remaining_df = train_test_split(
        eras_ft_data_df, test_size=FST_LAYER_TRAIN_RATIO)

    res = pd.DataFrame(False, index=remaining_df.index, columns=['fst_layer'])

    return fst_layer_df, res


def generate_data_subsets(subsets_dirname, data_df, subset_name, eras_fts):
    print("generate_data_subsets")

    eras = eras_fts['eras']
    fts = eras_fts['features']
    columns = ['era'] + fts + [TARGET_LABEL]
    print("eras: ", eras)
    print("fts: ", fts)
    eras_ft_data_df = data_df.loc[data_df['era'].isin(eras), columns]
    # print("eras_ft_data_df: ", eras_ft_data_df)

    write_file(subsets_dirname, subset_name, 'numerai_training_data.csv',
               eras_ft_data_df)

    eras_corr_matrix = correlation_matrix(eras_ft_data_df)
    write_file(subsets_dirname, subset_name, 'eras_corr.csv', eras_corr_matrix)

    train_df, test_df = train_test_split(eras_ft_data_df, test_size=TEST_RATIO)

    write_file(subsets_dirname, subset_name, 'training_data.csv', train_df)
    write_file(subsets_dirname, subset_name, 'test_data.csv', test_df)
    write_model_config(subset_name, 'model_config.json', eras)


def main():

    subsets_dirname = 'data_subsets_036'
    era_model_path = subsets_dirname + '/fst_layer_distribution.json'
    data_models = load_data_models(era_model_path)

    data_df = load_data(data_models['original_data_file'])
    data_remaining_df = pd.DataFrame()

    for name, eras_fts in data_models['subsets'].items():
        if 'data_subset' in name:
            print("name: ", name)
            print(" -> eras: ", eras_fts['eras'])
            print(" -> num ft: ", len(eras_fts['features']))
            fst_layer_data, remaining_data = split_fst_layer_data(
                data_df, eras_fts)
            data_remaining_df = pd.concat([data_remaining_df, remaining_data],
                                          axis=0)

            generate_data_subsets(subsets_dirname, fst_layer_data, name,
                                  eras_fts)

    data_df = pd.concat([data_df, data_remaining_df], axis=1)
    data_fst_layer_df = data_df['fst_layer'].fillna(True)
    data_fst_layer_df.index.name = 'id'

    with open('numerai_training_data_layer.csv', 'w') as f:
        data_fst_layer_df.to_csv(f)

    print("data_df: ", data_df)


if __name__ == '__main__':
    main()

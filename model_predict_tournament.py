import os
import sys
import json
import time
import math
import itertools
import numpy as np
import pandas as pd
import time

from reader_csv import ReaderCSV
from model_abstract import ModelType
from random_forest_model import RFModel
from xgboost_model import XGBModel
from neural_net_model import NeuralNetwork
from k_nn_model import K_NNModel


TOURNAMENT_NAME = "kazutsugi"
PREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"
COL_PRED = ['id', PREDICTION_NAME]
COL_PROBA_NAMES = ['proba_0.0', 'proba_0.25',
                   'proba_0.5', 'proba_0.75', 'proba_1.0']
TARGET_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
ERA_BATCH_SIZE = 10
# ERA_BATCH_SIZE = 2


def load_models_json(data_subsets_dirname, filepath):
    with open(filepath, 'r') as f:
        models = json.load(f)

    models_subsets = {subset['name']: {
        'corr_mat': load_corr_subset(data_subsets_dirname, subset['name']),
        'eras': subset['eras'],
        'features': subset['features']
    } for subset in models['subsets']}

    return models_subsets


def load_eras():
    data_filepath = 'numerai_tournament_data.csv'
    file_reader = ReaderCSV(data_filepath)
    eras_df = file_reader.read_csv(columns=['id', 'era']).set_index('id')

    return eras_df


def load_input_data_era(era_target):
    data_filepath = 'numerai_tournament_data.csv'
    file_reader = ReaderCSV(data_filepath)
    input_data = file_reader.read_csv_matching(
        'era', era_target).set_index('id')

    return input_data


def load_corr_subset(dirname, subset):
    corr_path = dirname + '/' + subset + '/eras_corr.csv'
    corr = ReaderCSV(corr_path).read_csv().set_index('corr_features')
    return corr


def load_model(subset_dirname, eModel, model_prefix=None):
    if eModel == ModelType.RandomForest:
        RF_model = RFModel(subset_dirname, debug=False)
        RF_model.load_model()
        return RF_model

    if eModel == ModelType.XGBoost:
        XGB_model = XGBModel(subset_dirname, debug=False)
        XGB_model.load_model()
        return XGB_model

    if eModel == ModelType.NeuralNetwork:
        NN_model = NeuralNetwork(subset_dirname, debug=False)
        NN_model.load_model()
        return NN_model

    if eModel == ModelType.K_NN:
        model_params = {"n_neighbors": model_prefix}
        k_nn_model = K_NNModel(
            subset_dirname, model_params=model_params, debug=False)
        k_nn_model.load_model()
        return k_nn_model


def model_proba(model, subset_ft, input_data):

    input_data_model_ft = input_data[subset_ft]

    prediction_proba = model.predict_proba(input_data_model_ft)

    columns_labels = [model.model_name + '_' +
                      proba_label for proba_label in COL_PROBA_NAMES]

    prediction_proba_df = pd.DataFrame(
        prediction_proba, input_data.index, columns=columns_labels)

    # print("prediction_proba_df.iloc[0]: ", prediction_proba_df.iloc[0])

    return prediction_proba_df


def snd_order_proba(subset_dir_path, model_types, input_data, subset_ft):
    models_proba_full = pd.DataFrame(dtype=np.float32)
    # models_proba_full = pd.DataFrame(index=input_data.index,
    #                                  columns=COL_PROBA_NAMES, dtype=np.float32)
    for col in models_proba_full.columns:
        models_proba_full[col].values[:] = 0.0

    for eModel in model_types:

        model_prefix_l = [5, 10, 30] if eModel == ModelType.K_NN else [None]

        for model_prefix in model_prefix_l:
            print("eModel: ", model_prefix, "-", eModel.name,)
            model = load_model(subset_dir_path, eModel, model_prefix)
            pred_proba = model_proba(model, subset_ft, input_data)
            # models_proba_full = models_proba_full.add(pred_proba)
            models_proba_full = pd.concat(
                [models_proba_full, pred_proba], axis=1)

    # For now, average proba between models
    # models_proba_full = models_proba_full.div(len(model_types))
    # print("models_proba_full: ", models_proba_full)

    return models_proba_full


def era_proba(data_subsets_dirname, input_data, subset_name, subset_ft):

    print("subset: ", subset_name)
    subset_dir_path = data_subsets_dirname + '/' + subset_name

    model_types = [ModelType.XGBoost, ModelType.RandomForest,
                   ModelType.NeuralNetwork, ModelType.K_NN]
    # model_types = [ModelType.K_NN]
    pred_proba = snd_order_proba(
        subset_dir_path, model_types, input_data, subset_ft)

    return pred_proba


def make_era_sub_proba(data_subsets_dirname, era, era_input_data, era_corr, subset_name, subset_data):
    # print("era_batch: ", era)
    # print("era_input_data: ", era_input_data)
    # print("era_corr: ", era_corr)
    # print("subset name: ", subset_name)
    # print("subset data: ", subset_data)

    sub_ft = subset_data['features']
    sub_era_ft_corr = era_corr.loc[era_corr.index.isin(sub_ft)][sub_ft]

    # ==========================
    # Need to check best formula for era (sub)sub_eradatasets similarity
    sub_era_corr_diff = (sub_era_ft_corr - subset_data['corr_mat'].values)
    sub_era_corr_dist = (sub_era_corr_diff / len(sub_ft)) ** 2

    sub_era_simi = math.sqrt(sub_era_corr_dist.values.sum())
    # ==========================

    sub_era = dict()
    sub_era['simi'] = sub_era_simi
    sub_era['proba'] = era_proba(
        data_subsets_dirname, era_input_data, subset_name, sub_ft)

    return sub_era


def make_pred_era(data_subsets_dirname, subsets, input_data, era):
    print("make_pred_era: ", era)
    era_input_data = input_data.loc[input_data['era'] == era].drop(
        'target_kazutsugi', axis=1)

    if era_input_data.empty:
        return None

    era_data_corr = era_input_data.corr()

    era_sub_proba = {name: make_era_sub_proba(data_subsets_dirname, era, era_input_data, era_data_corr, name, subset)
                     for name, subset in subsets.items()}

    # print('era_sub_proba: ', era_sub_proba)
    # Weighted arithmetic mean of all subsets model proba predictions
    total_sim = sum(sub_proba['simi']
                    for _, sub_proba in era_sub_proba.items())
    full_proba = sum(sub_proba['proba'] * sub_proba['simi']
                     for _, sub_proba in era_sub_proba.items()) / total_sim

    # era_pred_class = full_proba[COL_PROBA_NAMES].idxmax(axis=1)
    # era_pred_class = era_pred_class.map(lambda x: COL_PROBA_NAMES.index(x))
    # era_pred_class = era_pred_class.map(lambda x: TARGET_VALUES[x])

    # print("era_pred_class: ", era_pred_class)
    # return era_pred_class
    return full_proba


def make_pred(data_subsets_dirname, subsets, input_data, eras_batch):
    print("make_pred - eras_batch: ", eras_batch)

    era_batch_pred_class = [make_pred_era(
        data_subsets_dirname, subsets, input_data, era) for era in eras_batch]
    res = pd.concat(era_batch_pred_class)
    return res


def list_chunks(lst):
    for i in range(0, len(lst), ERA_BATCH_SIZE):
        yield lst[i:i + ERA_BATCH_SIZE]


def main():

    data_subsets_dirname = 'data_subsets_036'
    data_subsets_json_path = data_subsets_dirname + \
        '/' + data_subsets_dirname + '.json'

    data_types = ['validation', 'test', 'live']
    predictions_filepath = [data_subsets_dirname + '/predictions_tournament_validation.csv',
                            data_subsets_dirname + '/predictions_tournament_test.csv',
                            data_subsets_dirname + '/predictions_tournament_live.csv']

    data_types_fp = list(zip(data_types, predictions_filepath))
    file_write_header = {d_t: True for d_t, _ in data_types_fp}

    models_subsets = load_models_json(
        data_subsets_dirname, data_subsets_json_path)

    eras_df = load_eras()
    eras_list = eras_df['era'].drop_duplicates().values
    print("eras_df: ", eras_df)

    for era_b in list_chunks(eras_list):

        start_time = time.time()
        print("prediction for era batch: ", era_b)
        input_data = load_input_data_era(era_b)

        for data_t, fpath in data_types_fp:
            input_type_data = input_data.loc[input_data['data_type'] == data_t]

            if input_type_data.empty:
                continue

            pred_data = make_pred(data_subsets_dirname,
                                  models_subsets, input_type_data, era_b)

            write_mode = 'w' if file_write_header[data_t] else 'a'
            with open(fpath, write_mode) as f:
                pred_data.to_csv(
                    f, header=file_write_header[data_t], index=True)
                file_write_header[data_t] = False
        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()

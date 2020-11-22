import os
import json
import pandas as pd
import numpy as np
from common import *
from reader import ReaderCSV
from prediction import PredictionOperator
from models import ModelType


def load_json(fp):
    with open(fp, 'r') as f:
        json_data = json.load(f)

        return json_data


def load_data(fp):
    file_reader = ReaderCSV(fp)
    input_data = file_reader.read_csv().set_index('id')

    return input_data


def final_predict_layer(dirname, model_dict_pred, data_types_fp, num_layer):

    if num_layer not in LAYER_PRED_SUFFIX.keys():
        print("wrong layer number provided")
        return

    model_dict_k = 'fst_layer' if num_layer == 1 else 'snd_layer'
    layer_pred_descrb = model_dict_pred[model_dict_k]
    model_types = layer_pred_descrb['models_final']
    print("layer_pred_descrb: ", layer_pred_descrb)

    for data_t, fpath in data_types_fp:
        if data_t not in layer_pred_descrb.keys():
            continue

        if not os.path.exists(fpath):
            print("fst layer file not found at:", fpath)
            continue

        data_df = load_data(fpath)

        columns = layer_pred_descrb[data_t]

        result_vote = pd.DataFrame()

        proba_0_label = '_' + COL_PROBA_NAMES[0]
        model_col = [col.replace(proba_0_label,  '')
                     for col in columns if proba_0_label in col]

        for model in model_col:

            if model not in model_types:
                continue

            model_col = [col for col in data_df.columns if model in col]
            model_data = data_df[model_col]

            model_col_to_classes = dict(
                zip(model_data.columns, TARGET_CLASSES))
            model_data = model_data.rename(columns=model_col_to_classes)
            model_prediction = model_data.idxmax(axis=1)
            result_vote[model] = model_prediction

        class_col = {proba.replace('proba_', ''): list(filter(
            None, [col if proba in col else None for col in columns])) for proba in COL_PROBA_NAMES}

        arithmetic_mean = pd.DataFrame()
        # geometric_mean = pd.DataFrame()
        for target_class, columns in class_col.items():

            filtered_columns = [
                col for col in columns if col.startswith(tuple(model_types))]

            arithmetic_mean[target_class] = data_df[filtered_columns].mean(
                axis=1)
            # geometric_mean[target_class] = np.exp(
            #     np.log(data_df[columns].prod(axis=1))/data_df[columns].notna().sum(1))

        result_vote['arithmetic_mean'] = arithmetic_mean.idxmax(axis=1)

        pred_suffix = LAYER_PRED_SUFFIX[num_layer]
        predict_fp = dirname + '/' + FINAL_PREDICT_FILENAME + data_t + pred_suffix + '.csv'

        with open(predict_fp, 'w') as f:
            result_vote.to_csv(f, index=True)


def final_pred(strat_dir):

    model_dict_fp = strat_dir + '/' + MODEL_CONSTITUTION_FILENAME
    model_dict = load_json(model_dict_fp)

    # model_types = {'full': {'models': ['xgboost', 'rf', 'neural_net']},
    # model_types = {'fst': {'models': ['xgboost', 'rf', 'neural_net']},
    #                'snd': {'models': ['xgboost', 'rf', 'neural_net']}}
    model_dict_pred = model_dict['predictions']
    model_dict_pred['fst_layer']['models_final'] = ['neural_net']
    model_dict_pred['snd_layer']['models_final'] = [
        'xgboost', 'rf', 'neural_net']

    # FULL
    # full_dirname = dirname + '/full'
    # full_models_fp = full_dirname + '/full_models.json'
    # predictions_full_fp = [
    #     full_dirname + '/predictions_tournament_' + d_t + '_full.csv' for d_t in data_types]

    # fulll_data_types_fp = list(
    #     zip(data_types, predictions_full_fp))
    # final_predict_layer(dirname, full_models_fp, model_types['full']['models'],
    #                     fulll_data_types_fp, 0)

    # FST LAYER
    predictions_fst_layer_fp = [
        strat_dir + '/predictions_tournament_' + d_t + '_fst_layer.csv' for d_t in PREDICTION_TYPES]

    fst_layer_data_types_fp = list(
        zip(PREDICTION_TYPES, predictions_fst_layer_fp))
    final_predict_layer(strat_dir, model_dict_pred, fst_layer_data_types_fp, 1)

    # SND LAYER
    predictions_snd_layer_fp = [
        strat_dir + '/predictions_tournament_' + d_t + '_snd_layer.csv' for d_t in PREDICTION_TYPES]

    snd_layer_data_types_fp = list(
        zip(PREDICTION_TYPES, predictions_snd_layer_fp))
    final_predict_layer(strat_dir, model_dict_pred, snd_layer_data_types_fp, 2)

    with open(model_dict_fp, 'w') as fp:
        json.dump(model_dict, fp, indent=4)

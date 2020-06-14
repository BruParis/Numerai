import os
import json
import pandas as pd
import numpy as np
from reader_csv import ReaderCSV
from prediction_operator import PredictionOperator
from model_abstract import ModelType

LABEL_CLASSES = ['0.0', '0.25', '0.5', '0.75', '1.0']

COL_PROBA_NAMES = ['proba_0.0', 'proba_0.25',
                   'proba_0.5', 'proba_0.75', 'proba_1.0']

LAYER_PRED_SUFFIX = {1: '_fst', 2: '_snd'}


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def load_data(filepath):
    file_reader = ReaderCSV(filepath)
    input_data = file_reader.read_csv().set_index('id')

    return input_data


def final_predict_layer(dirname, layer_distrib_filepath, model_types, data_types_fp, num_layer):

    if num_layer not in LAYER_PRED_SUFFIX.keys():
        print("wrong layer number provided")
        return

    layer_distrib = load_json(layer_distrib_filepath)
    layer_pred_descrb = layer_distrib['prediction']
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

            model_col_to_classes = dict(zip(model_data.columns, LABEL_CLASSES))
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
        # result_vote['geometric_mean'] = geometric_mean.idxmax(axis=1)

        predict_file_suffix = LAYER_PRED_SUFFIX[num_layer]
        predict_filepath = dirname + '/final_predict_' + \
            data_t + predict_file_suffix + '.csv'

        with open(predict_filepath, 'w') as f:
            result_vote.to_csv(f, index=True)


def main():

    dirname = "data_subsets_036"
    snd_layer_dirname = dirname + '/snd_layer'

    data_types = ['validation', 'test', 'live']

    model_types_fp = dirname + '/final_predict_scores.json'
    # model_types = load_json(model_types_fp)
    # for key, _ in model_types.items():
    #     if key
    model_types = {'fst': {'models': ['xgboost', 'rf', 'neural_net']},
                   'snd': {'models': ['xgboost', 'rf', 'neural_net']}}

    with open(model_types_fp, 'w') as fp:
        json.dump(model_types, fp, indent=4)

    # FST LAYER
    fst_layer_distrib_filepath = dirname + '/fst_layer_distribution.json'
    predictions_fst_layer_filepath = [
        dirname + '/predictions_tournament_' + d_t + '_fst_layer.csv' for d_t in data_types]

    fst_layer_data_types_fp = list(
        zip(data_types, predictions_fst_layer_filepath))
    final_predict_layer(dirname, fst_layer_distrib_filepath, model_types['fst']['models'],
                        fst_layer_data_types_fp, 1)

    # SND LAYER
    snd_layer_dirname = dirname + '/snd_layer'
    snd_layer_distrib_filepath = snd_layer_dirname + '/snd_layer_models.json'
    predictions_snd_layer_filepath = [
        snd_layer_dirname + '/predictions_tournament_' + d_t + '_snd_layer.csv' for d_t in data_types]

    snd_layer_data_types_fp = list(
        zip(data_types, predictions_snd_layer_filepath))
    final_predict_layer(dirname, snd_layer_distrib_filepath, model_types['snd']['models'],
                        snd_layer_data_types_fp, 2)


if __name__ == '__main__':
    main()

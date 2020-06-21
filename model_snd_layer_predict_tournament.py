import os
import sys
import json
import math
import pandas as pd
import numpy as np
import time

import model_itf

from pool_map import pool_map
from reader_csv import ReaderCSV
from model_abstract import ModelType
from prediction_operator import PredictionOperator


TARGET_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
ERA_BATCH_SIZE = 32
# ERA_BATCH_SIZE = 2


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def load_eras():
    data_filepath = 'numerai_tournament_data.csv'
    file_reader = ReaderCSV(data_filepath)
    eras_df = file_reader.read_csv(columns=['id', 'era']).set_index('id')

    return eras_df


def load_data(data_filepath):
    file_reader = ReaderCSV(data_filepath)
    input_data = file_reader.read_csv().set_index('id')

    return input_data


def load_matching_data(data_filepath, era_target):
    file_reader = ReaderCSV(data_filepath)
    input_data = file_reader.read_csv_matching(
        'era', era_target).set_index('id')

    return input_data


def list_chunks(lst):
    for i in range(0, len(lst), ERA_BATCH_SIZE):
        yield lst[i:i + ERA_BATCH_SIZE]


def main():

    try:
        data_types = [sys.argv[1]]
    except IndexError:
        print(
            "Usage: validation_score.py <data_type : ['validation', 'test', 'live']>")
        print("-----> default value : snd")
        data_types = ['validation', 'test', 'live']

    data_subsets_dirname = 'data_subsets_036'
    snd_layer_dirname = data_subsets_dirname + '/snd_layer'
    snd_layer_distrib_filename = 'snd_layer_models.json'

    numerai_training_data = 'numerai_training_data.csv'

    predictions_snd_layer_filepath = [snd_layer_dirname +
                                      '/predictions_tournament_' + d_t + '_snd_layer.csv' for d_t in data_types]

    data_types_fp = list(zip(data_types, predictions_snd_layer_filepath))
    file_write_header = {d_t: True for d_t, _ in data_types_fp}

    # models_subsets = load_models_json(
    #     data_subsets_dirname, data_subsets_json_path)

    # model_types = [ModelType.NeuralNetwork]
    model_types = [ModelType.XGBoost, ModelType.RandomForest,
                   ModelType.NeuralNetwork]  # , ModelType.K_NN]

    bLoadFstLayer = True
    snd_layer_pred_descr = {}
    if bLoadFstLayer:

        predictions_fst_layer_filepath = [data_subsets_dirname +
                                          '/predictions_tournament_' + d_t + '_fst_layer.csv' for d_t in data_types]

        data_layers_fp = list(
            zip(data_types, predictions_fst_layer_filepath, predictions_snd_layer_filepath))

        for data_type, fst_layer_path, snd_layer_path in data_layers_fp:

            if not os.path.exists(fst_layer_path):
                print("fst layer file not found at:", fst_layer_path)
                continue

            fst_layer_data = load_data(fst_layer_path)

            print("fst_layer_path: ", fst_layer_path)
            print("snd_layer_path: ", snd_layer_path)
            print("fst_layer_data: ", fst_layer_data)

            print("snd_layer_distrib_filename: ", snd_layer_distrib_filename)

            pred_op = PredictionOperator(
                snd_layer_dirname, snd_layer_distrib_filename, model_types, bMultiProcess=False)

            snd_layer_pred_data = pred_op.make_full_predict(fst_layer_data)

            # stitch eras back
            snd_layer_pred_data = pd.concat(
                [snd_layer_pred_data, fst_layer_data['era']], axis=1)

            snd_layer_pred_descr[data_type] = snd_layer_pred_data.columns.values.tolist(
            )

            with open(snd_layer_path, 'w') as f:
                snd_layer_pred_data.to_csv(f, index=True)

    else:
        # ************************
        # NOT FUNCTIONNAL FOR NOW
        # ************************
        fst_layer_distrib_filename = '/fst_layer_distribution.json'

        eras_df = load_eras()
        eras_list = eras_df['era'].drop_duplicates().values
        print("eras_df: ", eras_list)

        eras_list = eras_list

        for era_b in list_chunks(eras_list):

            start_time = time.time()
            print("prediction for era batch: ", era_b)
            input_data = load_matching_data(numerai_training_data, era_b)

            for era in era_b:
                print("era: ", era)
                input_data_era = input_data.loc[input_data['era'] == era]

                for data_t, fpath in data_types_fp:
                    input_type_data = input_data_era.loc[input_data_era['data_type'] == data_t]

                    if input_type_data.empty:
                        continue

                    pred_op = PredictionOperator(data_subsets_dirname, fst_layer_distrib_filename, model_types,
                                                 bMultiProcess=False)

                    pred_data = pred_op.make_fst_layer_predict(input_type_data)

                    print("pred_data: ", pred_data)

                    write_mode = 'w' if file_write_header[data_t] else 'a'
                    with open(fpath, write_mode) as f:
                        pred_data.to_csv(
                            f, header=file_write_header[data_t], index=True)
                        file_write_header[data_t] = False

            print("--- %s seconds ---" % (time.time() - start_time))

    snd_layer_distrib_filpath = snd_layer_dirname + '/' + snd_layer_distrib_filename
    snd_layer_distrib = load_json(snd_layer_distrib_filpath)

    snd_layer_distrib['prediction'] = snd_layer_pred_descr
    print("snd_layer_distrib: ", snd_layer_distrib)

    with open(snd_layer_distrib_filpath, 'w') as fp:
        json.dump(snd_layer_distrib, fp, indent=4)


if __name__ == '__main__':
    main()

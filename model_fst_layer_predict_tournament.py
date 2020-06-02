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
#ERA_BATCH_SIZE = 2


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def load_eras():
    data_filepath = 'numerai_tournament_data.csv'
    file_reader = ReaderCSV(data_filepath)
    eras_df = file_reader.read_csv(columns=['id', 'era']).set_index('id')

    return eras_df


def load_input_data(era_target):
    data_filepath = 'numerai_tournament_data.csv'
    file_reader = ReaderCSV(data_filepath)
    input_data = file_reader.read_csv_matching(
        'era', era_target).set_index('id')

    return input_data


def list_chunks(lst):
    for i in range(0, len(lst), ERA_BATCH_SIZE):
        yield lst[i:i + ERA_BATCH_SIZE]


def main():

    data_subsets_dirname = 'data_subsets_036'
    fst_layer_distrib_filename = '/fst_layer_distribution.json'

    data_types = ['validation', 'test', 'live']
    predictions_filepath = [data_subsets_dirname + '/predictions_tournament_validation_fst_layer.csv',
                            data_subsets_dirname + '/predictions_tournament_test_fst_layer.csv',
                            data_subsets_dirname + '/predictions_tournament_live_fst_layer.csv']

    data_types_fp = list(zip(data_types, predictions_filepath))
    file_write_header = {d_t: True for d_t, _ in data_types_fp}

    # models_subsets = load_models_json(
    #     data_subsets_dirname, data_subsets_json_path)

    eras_df = load_eras()
    eras_list = eras_df['era'].drop_duplicates().values
    print("eras_df: ", eras_list)

    eras_list = eras_list

    model_types = [ModelType.XGBoost,
                   ModelType.RandomForest, ModelType.NeuralNetwork]

    fst_layer_pred_descr = {}
    for era_b in list_chunks(eras_list):

        start_time = time.time()
        print("prediction for era batch: ", era_b)
        input_data = load_input_data(era_b)

        for era in era_b:
            print("era: ", era)
            input_data_era = input_data.loc[input_data['era'] == era]

            for data_t, fpath in data_types_fp:

                input_type_data = input_data_era.loc[input_data_era['data_type'] == data_t]

                if input_type_data.empty:
                    continue

                pred_op = PredictionOperator(data_subsets_dirname, fst_layer_distrib_filename, model_types,
                                             bMultiProcess=True)

                pred_data = pred_op.make_fst_layer_predict(input_type_data)

                # stitch eras back
                pred_data = pd.concat(
                    [pred_data, input_type_data['era']], axis=1)

                if file_write_header[data_t]:
                    fst_layer_pred_descr[data_t] = pred_data.columns.values.tolist(
                    )

                write_mode = 'w' if file_write_header[data_t] else 'a'
                with open(fpath, write_mode) as f:
                    pred_data.to_csv(
                        f, header=file_write_header[data_t], index=True)
                    file_write_header[data_t] = False

        print("--- %s seconds ---" % (time.time() - start_time))

    fst_layer_distrib_filpath = data_subsets_dirname + fst_layer_distrib_filename
    fst_layer_distrib = load_json(fst_layer_distrib_filpath)
    fst_layer_distrib['prediction'] = fst_layer_pred_descr

    with open(fst_layer_distrib_filpath, 'w') as fp:
        json.dump(fst_layer_distrib, fp, indent=4)


if __name__ == '__main__':
    main()

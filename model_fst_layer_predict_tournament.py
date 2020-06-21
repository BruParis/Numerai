import os
import sys
import json
import math
import argparse
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


def load_eras_data_type():
    data_filepath = 'numerai_tournament_data.csv'
    file_reader = ReaderCSV(data_filepath)
    eras_df = file_reader.read_csv(
        columns=['id', 'era', 'data_type']).set_index('id')

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


def usage():
    print("Usage: validation_score.py <data_type> <full>")
    print("     -> data_type value : ['validation', 'test', 'live']")
    print("     -> <full> : None -> use subsets; 'full' full dataset")
    return


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("data_type",
                        help="<data_type> used for prediction",
                        nargs='*',
                        default=['validation', 'test', 'live'])
    parser.add_argument("-f", "--full", action='store_true',
                        help="use full dataset")
    args = parser.parse_args()
    data_types = args.data_type
    bFull = args.full

    print('data_types: ', data_types)
    print('use full dataset: ', bFull)

    data_subsets_dirname = 'data_subsets_036/full' if bFull else 'data_subsets_036'
    models_filename = '/full_models.json' if bFull else '/fst_layer_distribution.json'
    prediction_suffix = '_full.csv' if bFull else '_fst_layer.csv'

    predictions_filepath = [data_subsets_dirname +
                            '/predictions_tournament_' + d_t + prediction_suffix for d_t in data_types]

    data_types_fp = list(zip(data_types, predictions_filepath))
    file_write_header = {d_t: True for d_t, _ in data_types_fp}

    eras_type_df = load_eras_data_type()

    model_types = [ModelType.XGBoost,
                   ModelType.RandomForest, ModelType.NeuralNetwork]

    prediction_descr = {}
    for data_t, fpath in data_types_fp:

        eras_df = eras_type_df.loc[eras_type_df['data_type'] == data_t]
        eras_list = eras_df['era'].drop_duplicates().values
        print("eras_df: ", eras_list)

        eras_list = eras_list

        for era_b in list_chunks(eras_list):

            start_time = time.time()
            print("prediction for era batch: ", era_b)
            input_data = load_input_data(era_b)

            for era in era_b:
                print("era: ", era)
                input_data_era = input_data.loc[input_data['era'] == era]

                input_type_data = input_data_era.loc[input_data_era['data_type'] == data_t]

                if input_type_data.empty:
                    continue

                pred_op = PredictionOperator(data_subsets_dirname, models_filename, model_types,
                                             bMultiProcess=True)

                pred_data = pred_op.make_full_predict(
                    input_type_data) if bFull else pred_op.make_fst_layer_predict(input_type_data)

                # stitch eras back
                pred_data = pd.concat(
                    [pred_data, input_type_data['era']], axis=1)

                if file_write_header[data_t]:
                    prediction_descr[data_t] = pred_data.columns.values.tolist(
                    )

                write_mode = 'w' if file_write_header[data_t] else 'a'
                with open(fpath, write_mode) as f:
                    pred_data.to_csv(
                        f, header=file_write_header[data_t], index=True)
                    file_write_header[data_t] = False

        print("--- %s seconds ---" % (time.time() - start_time))

    models_descr_filpath = data_subsets_dirname + models_filename
    models_descr = load_json(models_descr_filpath)
    models_descr['prediction'] = prediction_descr

    with open(models_descr_filpath, 'w') as fp:
        json.dump(models_descr, fp, indent=4)


if __name__ == '__main__':
    main()

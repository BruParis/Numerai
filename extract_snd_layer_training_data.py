import os
import errno
import json
import pandas as pd
from reader_csv import ReaderCSV

from model_abstract import Model, ModelType
from prediction_operator import PredictionOperator


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def load_data_matching_id(data_filename, list_id):
    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv_filter_id(list_id).set_index("id")

    return data_df


def load_data_snd_layer(data_filename):
    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv().set_index("id")

    print("data_df: ", data_df)
    data_df = data_df[data_df['fst_layer'] == False]

    return data_df


def write_file(filepath, data):

    print("write file: ", filepath)
    with open(filepath, 'w') as f:
        data.to_csv(f)


def main():

    data_subsets_dirname = 'data_subsets_036'
    snd_layer_dirname = 'data_subsets_036/snd_layer'

    try:
        os.makedirs(snd_layer_dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Error with : make dir ", snd_layer_dirname)
            exit(1)

    fst_layer_filename = 'numerai_training_data_layer.csv'
    fst_layer_distrib_filename = 'fst_layer_distribution.json'

    snd_layer_data_id = load_data_snd_layer(fst_layer_filename)

    print("snd_layer_data_id.index: ", snd_layer_data_id.index)

    data_filename = 'numerai_training_data.csv'
    snd_layer_data = load_data_matching_id(
        data_filename, snd_layer_data_id.index)

    print("snd_layer_data: ", snd_layer_data)

    fst_layer_model_types = [ModelType.XGBoost,
                             ModelType.RandomForest, ModelType.NeuralNetwork]

    pred_op_fst_layer = PredictionOperator(
        data_subsets_dirname, fst_layer_distrib_filename, fst_layer_model_types, bMultiProcess=False)

    eras_list = snd_layer_data['era'].drop_duplicates().values
    print("eras_df: ", eras_list)

    pred_fst_layer_data = []
    for era in eras_list:
        print('era: ', era)

        era_snd_layer_data = snd_layer_data.loc[snd_layer_data['era'] == era]

        era_pred_data = pred_op_fst_layer.make_fst_layer_predict(
            era_snd_layer_data)
        era_pred_data['era'] = era

        pred_fst_layer_data.append(era_pred_data)

    pred_fst_layer_full_data = pd.concat(pred_fst_layer_data)

    snd_layer_training_data_filepath = snd_layer_dirname + \
        '/snd_layer_training_data.csv'
    write_file(snd_layer_training_data_filepath, pred_fst_layer_full_data)


if __name__ == '__main__':
    main()

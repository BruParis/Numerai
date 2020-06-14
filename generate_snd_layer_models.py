from sklearn.model_selection import train_test_split
import os
import sys
import json
import errno
import numpy as np
import pandas as pd
import itertools
import json

from pool_map import pool_map

from reader_csv import ReaderCSV
from model_abstract import Model, ModelType
from model_generator import ModelGenerator

TARGET_LABEL = 'target_kazutsugi'
TEST_RATIO = 0.20

# Random Forest
# FST ESTIMATION FOR PARAMETERS BOUNDS :
# n_est : < 268;340
# m_depth :  < 28;32
# min_splt : >= 10
# min_leaf : >= 4


def load_data(data_filename):
    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv().set_index("id")

    return data_df


def load_data_filter_id(data_filename, list_id, columns=None):

    print("list_id: ", list_id)
    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv_filter_id(
        list_id, columns=columns).set_index("id")

    return data_df


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def make_model_params(eModel, model_prefix=None):
    if eModel == ModelType.RandomForest:
        n_est = [250]  # np.linspace(start=180, stop=250, num=7)
        max_d = [10]  # np.linspace(10, 20, num=5)
        model_params_array = map(lambda x: {'n_estimators': int(
            x[0]), 'max_depth': int(x[1])}, itertools.product(*[n_est, max_d]))

        return model_params_array

    if eModel == ModelType.XGBoost:
        n_est = [180]  # np.linspace(start=180, stop=250, num=7)
        max_d = [5]  # np.linspace(5, 20, num=5)
        # eta = learning_rate
        eta = [1.0]  # np.logspace(start=(-1.0), stop=0.0, base=10.0, num=5)

        model_params_array = map(lambda x: {'n_estimators': int(x[0]), 'max_depth': int(
            x[1]), 'learning_rate': float(x[2])}, itertools.product(*[n_est, max_d, eta]))

        return model_params_array

    if eModel == ModelType.NeuralNetwork:
        num_layers = [1]  # np.linspace(start=1, stop=4, num=1)
        layer_size_factor = [0.5]  # [0.33, 0.5, 0.66]
        model_params_array = map(
            lambda x: {'num_layers': int(x[0]), 'size_factor': float(
                x[1])}, itertools.product(*[num_layers, layer_size_factor]))

        return model_params_array

    if eModel == ModelType.K_NN:
        leaf_size = [20]  # np.linspace(start=20, stop=50, num=1)
        minkowski_dist = [1]  # [1, 2, 3, 4]
        model_params_array = map(
            lambda x: {'n_neighbors': int(model_prefix), 'leaf_size': int(x[0]), 'minkowski_dist': int(
                x[1])}, itertools.product(*[leaf_size, minkowski_dist]))

        return model_params_array


def make_model_prefix(eModel):
    if eModel is not ModelType.K_NN:
        return [None]

    return [5, 10, 30]


def snd_layer_model_build(snd_layer_dirname, full_train_data, bSaveModel=False, bMetrics=False, model_debug=False):

    train_data, test_data = train_test_split(
        full_train_data, test_size=TEST_RATIO)

    model_types = [ModelType.XGBoost, ModelType.RandomForest,
                   ModelType.NeuralNetwork]  # , ModelType.K_NN]

    model_generator = ModelGenerator(snd_layer_dirname, train_data, test_data)

    model_l = []
    for model_type in model_types:
        for model_prefix in make_model_prefix(model_type):

            model_generator.start_model_type(
                model_type, model_prefix, bMetrics)

            model_params_array = make_model_params(model_type, model_prefix)

            best_ll = sys.float_info.max
            for model_params in model_params_array:

                print("generate model")
                model, model_dict = model_generator.generate_model(
                    model_params, model_debug)
                print("model: ", model)
                print("model_dict: ", model_dict)

                log_loss, _ = model_dict['log_loss'], model_dict['accuracy_score']

                if bSaveModel and (log_loss < best_ll):
                    best_ll = log_loss
                    filepath, configpath = model.save_model()
                    model_dict['model_filepath'] = filepath
                    model_dict['config_filepath'] = configpath
                    model_l.append(model_dict)

    res = {'models': model_l}
    return res


def main():

    snd_layer_dirname = 'data_subsets_036/snd_layer'

    bDebug = True
    bMetrics = False
    bSaveModel = True

    # bSaveModelDict = False

    training_data_filename = 'numerai_training_data.csv'

    snd_layer_training_data_filename = snd_layer_dirname + '/snd_layer_training_data.csv'
    snd_layer_training_data = load_data(snd_layer_training_data_filename)

    snd_layer_training_data_target = load_data_filter_id(
        training_data_filename, snd_layer_training_data.index, ['id', TARGET_LABEL])

    print("snd_layer_training_data: ", snd_layer_training_data)
    print("snd_layer_training_data_target: ", snd_layer_training_data_target)

    snd_layer_training_data = pd.concat(
        [snd_layer_training_data, snd_layer_training_data_target], axis=1)

    print("snd_layer_training_data: ", snd_layer_training_data)

    model_dict_sub_l = snd_layer_model_build(snd_layer_dirname, snd_layer_training_data,
                                             bSaveModel=bSaveModel,
                                             bMetrics=bMetrics,
                                             model_debug=bDebug)

    print("model building done")

    # or bSaveModelDict:
    if bSaveModel:
        snd_layer_models_fp = snd_layer_dirname + '/snd_layer_models.json'

        with open(snd_layer_models_fp, 'w') as fp:
            json.dump(model_dict_sub_l, fp, indent=4)


if __name__ == '__main__':
    main()

from sklearn.model_selection import train_test_split

import os
import sys
import json
import numpy as np
import itertools

from pool_map import pool_map

from reader_csv import ReaderCSV
from model_abstract import Model, ModelType
from model_generator import ModelGenerator

TEST_RATIO = 0.20

# Random Forest
# FST ESTIMATION FOR PARAMTERS BOUNDS :
# n_est : < 268;340
# m_depth :  < 28;32
# min_splt : >= 10
# min_leaf : >= 4


def load_data_matching(data_filename, filter_col, filter_values):
    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv(filter_col, filter_values).set_index("id")

    return data_df


def load_data_id(data_filename, id_values):
    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv(
        skip_lambda=lambda x: x.isin(id_values)).set_index("id")

    return data_df


def make_model_params(eModel, model_prefix=None):
    if eModel == ModelType.RandomForest:
        n_est = [220]  # np.linspace(start=180, stop=250, num=20)
        max_d = [10]  # np.linspace(10, 20, num=5)
        model_params_array = map(lambda x: {'n_estimators': int(
            x[0]), 'max_depth': int(x[1])}, itertools.product(*[n_est, max_d]))

        return model_params_array

    if eModel == ModelType.XGBoost:
        n_est = [220]  # np.linspace(start=180, stop=250, num=10)
        max_d = [10]  # np.linspace(10, 20, num=5)
        # eta = learning_rate
        eta = [1.0]  # np.logspace(start=(-1.0), stop=0.0, base=10.0, num=5)

        model_params_array = map(lambda x: {'n_estimators': int(x[0]), 'max_depth': int(
            x[1]), 'learning_rate': float(x[2])}, itertools.product(*[n_est, max_d, eta]))

        return model_params_array

    if eModel == ModelType.NeuralNetwork:
        num_layers = [1]  # np.linspace(start=1, stop=4, num=1)
        layer_size_factor = [0.66]
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


def build_models(dir_path, snd_layer_data, bSaveModel=False, bMetrics=False, model_debug=False):

    train_df, test_df = train_test_split(
        snd_layer_data, test_size=TEST_RATIO)

    # bMultiProc = False
    # if bMultiProc:
    # Parallelism only shows a slight speed improvment
    # -> not enough memory for mutliple thread w/ full dataset (?)
    # model_metrics_arg = list(zip(itertools.repeat(dir_path), itertools.repeat(
    #     metrics_filename), itertools.repeat(train_data), itertools.repeat(test_data), model_params_array))
    # pool_map(model_itf.generate_model, model_metrics_arg)
    # else:

    # model_types = ModelType
    # model_types = [ModelType.K_NN]
    # model_types = [ModelType.RandomForest]
    model_types = [ModelType.XGBoost, ModelType.RandomForest,
                   ModelType.NeuralNetwork, ModelType.K_NN]

    model_generator = ModelGenerator(
        dir_path, train_data, test_data)

    model_l = []
    for model_type in model_types:
        for model_prefix in make_model_prefix(model_type):

            model_generator.start_model_type(
                model_type, model_prefix, bMetrics)

            model_params_array = make_model_params(model_type, model_prefix)

            best_ll = sys.float_info.max
            for model_params in model_params_array:

                model, model_dict = model_generator.generate_model(
                    model_params, model_debug)

                log_loss, _ = model_dict['log_loss'], model_dict['accuracy_score']

                if bSaveModel and (log_loss < best_ll):
                    best_ll = log_loss
                    model.save_model()
                    model_l.append(model_dict)

    return model_l


def main():

    dirname = 'data_subsets_036'
    _, subdirs, _ = next(os.walk(dirname))
    print("subdirs: ", subdirs)

    snd_layer_id = load_data_matching(
        'numerai_training_data_layer.csv', 'fst_layer', False)
    snd_layer_data = load_data_id(
        'numerai_training_data.csv', snd_layer_id.index)

    bDebug = False
    bMetrics = False
    bSaveModel = True

    bMultiProc = False
    bSaveModelDict = True

    model_dict_sub_l = build_models(
        sub_dir_path, snd_layer_data, bSaveModel, bMetrics, bDebug)

    if bSaveModelDict:
        model_dict_filepath = dirname + '/fst_layer_model.json'
        with open(model_dict_filepath, 'w') as fp:
            json.dump(model_dict_sub_l, fp, indent=4)


if __name__ == '__main__':
    main()

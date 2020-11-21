from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
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

ERA_BATCH_SIZE = 8


def load_data(data_filepath):
    file_reader = ReaderCSV(data_filepath)
    input_data = file_reader.read_csv().set_index('id')
    input_data = input_data.drop(["data_type"], axis=1)

    return input_data


def load_matching_data(data_filepath, era_target):
    file_reader = ReaderCSV(data_filepath)
    input_data = file_reader.read_csv_matching(
        'era', era_target).set_index('id')
    input_data = input_data.drop(["data_type"], axis=1)

    return input_data


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


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


def list_chunks(lst):
    for i in range(0, len(lst), ERA_BATCH_SIZE):
        yield lst[i:i + ERA_BATCH_SIZE]


def model_build(dirname, bSaveModel=False, bMetrics=False, model_debug=False):

    data_filename = 'numerai_training_data.csv'

    model_types = [ModelType.RandomForest, ModelType.XGBoost,
                   ModelType.NeuralNetwork, ModelType.K_NN]

    model_generator = ModelGenerator(dirname)

    model_l = []

    for model_type in model_types:
        for model_prefix in make_model_prefix(model_type):

            model_generator.start_model_type(
                model_type, model_prefix, bMetrics)

            model_params_array = make_model_params(
                model_type, model_prefix)

            best_ll = sys.float_info.max
            for model_params in model_params_array:
                model_generator.generate_model(model_params, model_debug)

                input_data = load_data(data_filename)

                input_data = shuffle(input_data)

                train_data, test_data = train_test_split(
                    input_data, test_size=TEST_RATIO)

                print("generate model")
                print("train_data: ", train_data)
                model, model_dict = model_generator.build_evaluate_model(
                    train_data, test_data)
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

    bDebug = True
    bMetrics = False
    bSaveModel = True

    # bSaveModelDict = False

    dirname = 'data_subsets_036/full'

    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Error with : make dir ", dirname)
            exit(1)

    model_dict_sub_l = model_build(dirname, bSaveModel=bSaveModel,
                                   bMetrics=bMetrics,
                                   model_debug=bDebug)

    print("model building done")

    # or bSaveModelDict:
    if bSaveModel:
        full_models_fp = dirname + '/full_models.json'

        with open(full_models_fp, 'w') as fp:
            json.dump(model_dict_sub_l, fp, indent=4)


if __name__ == '__main__':
    main()

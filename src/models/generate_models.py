import os
import sys
import json
import numpy as np
import pandas as pd
import itertools
import json
import time
from sklearn.model_selection import train_test_split

from common import *
from threadpool import pool_map
from reader import ReaderCSV
from .model_abstract import Model, ModelType
from .model_generator import ModelGenerator

# Random Forest
# FST ESTIMATION FOR PARAMTERS BOUNDS :
# n_est : < 268;340
# m_depth :  < 28;32
# min_splt : >= 10
# min_leaf : >= 4


def load_data(data_filename):
    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv().set_index("id")

    return data_df


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def make_fst_layer_model_params(eModel, model_prefix=None):
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


def make_snd_layer_model_params(eModel, model_prefix=None):
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
        layer_size_factor = [1]  # [0.33, 0.5, 0.66]
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


def cl_model_build(dirname, cl_dirname, bSaveModel=False, bMetrics=False, model_debug=False):

    cl_dirpath = dirname + '/' + cl_dirname
    train_filepath = cl_dirpath + '/' + 'training_data.csv'
    test_filepath = cl_dirpath + '/' + 'test_data.csv'

    train_data = load_data(train_filepath)
    test_data = load_data(test_filepath)

    # bMultiProc = False
    # if bMultiProc:
    # Parallelism only shows a slight speed improvment
    # -> not enough memory for mutliple thread w/ full dataset (?)
    # model_metrics_arg = list(zip(itertools.repeat(cl_dirname), itertools.repeat(
    #     metrics_filename), itertools.repeat(train_data), itertools.repeat(test_data), model_params_array))
    # pool_map(model_itf.generate_model, model_metrics_arg)
    # else:

    # model_types = ModelType
    # model_types = [ModelType.K_NN]
    # model_types = [ModelType.RandomForest]
    model_types = [ModelType.NeuralNetwork]
    # model_types = [ModelType.XGBoost, ModelType.RandomForest,
    #                ModelType.NeuralNetwork]  # , ModelType.K_NN]

    model_generator = ModelGenerator(cl_dirpath)

    model_l = []
    for model_type in model_types:
        for model_prefix in make_model_prefix(model_type):

            model_generator.start_model_type(
                model_type, model_prefix, bMetrics)

            model_params_array = make_fst_layer_model_params(
                model_type, model_prefix)

            best_ll = sys.float_info.max
            for model_params in model_params_array:

                model_generator.generate_model(model_params, model_debug)
                model, model_dict = model_generator.build_evaluate_model(
                    train_data, test_data)

                log_loss, _ = model_dict['log_loss'], model_dict['accuracy_score']

                if bSaveModel and (log_loss < best_ll):
                    best_ll = log_loss
                    filepath, configpath = model.save_model()
                    model_dict['model_filepath'] = filepath
                    model_dict['config_filepath'] = configpath
                    model_l.append(model_dict)

    return cl_dirname, model_l


def generate_fst_layer_model(dirname, cl_dirname_l, bDebug, bMetrics, bSaveModel,
                             bMultiProc, bSaveModelDict):
    start_time = time.time()

    # Seems there is a pb with multiprocess (mult. proc w/ same dataframe?)
    model_dict_l = {}
    if bMultiProc:
        models_build_arg = list(zip(itertools.repeat(dirname), cl_dirname_l, itertools.repeat(
            bSaveModel), itertools.repeat(bMetrics), itertools.repeat(bDebug)))
        cl_dir_model_l = pool_map(
            cl_model_build, models_build_arg, collect=True, arg_tuple=True)

        model_dict_l = dict(cl_dir_model_l)
    else:
        for cl_dir in cl_dirname_l:
            _, model_dict_cl_l = cl_model_build(
                dirname, cl_dir, bSaveModel, bMetrics, bDebug)

            model_dict_l[cl_dir] = model_dict_cl_l

    print("model building done")
    print("--- %s seconds ---" % (time.time() - start_time))

    if bSaveModel or bSaveModelDict:
        model_c_filepath = dirname + '/' + MODEL_CONSTITUTION_FILENAME

        cl_dict = load_json(model_c_filepath)
        print("model_c_filepath: ", model_c_filepath)
        print("cl_dict: ", cl_dict)
        print("model_dict_l.keys(): ", model_dict_l.keys())

        for cl_name, cl_desc in cl_dict["clusters"].items():

            if cl_name in model_dict_l.keys():
                cl_desc['models'] = model_dict_l[cl_name]

        with open(model_c_filepath, 'w') as fp:
            json.dump(cl_dict, fp, indent=4)


def load_data_filter_id(data_filename, list_id, columns=None):

    print("list_id: ", list_id)
    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv_filter_id(
        list_id, columns=columns).set_index("id")

    return data_df


def snd_layer_model_build(snd_layer_dirname, full_train_data, bSaveModel=False,
                          bMetrics=False, model_debug=False):

    train_data, test_data = train_test_split(
        full_train_data, test_size=TEST_RATIO)

    model_types = [ModelType.XGBoost, ModelType.RandomForest,
                   ModelType.NeuralNetwork]  # , ModelType.K_NN]

    model_generator = ModelGenerator(snd_layer_dirname)

    model_l = []
    for model_type in model_types:
        for model_prefix in make_model_prefix(model_type):

            model_generator.start_model_type(
                model_type, model_prefix, bMetrics)

            model_params_array = make_snd_layer_model_params(
                model_type, model_prefix)

            best_ll = sys.float_info.max
            for model_params in model_params_array:

                print("generate model")
                model_generator.generate_model(model_params, model_debug)
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

    return model_l


def generate_snd_layer_model(dirname, bDebug, bMetrics, bSaveModel):
    snd_layer_dirname = dirname + '/' + SND_LAYER_DIRNAME

    snd_layer_training_data_filename = snd_layer_dirname + '/' + SND_LAYER_FILENAME
    snd_layer_training_data = load_data(snd_layer_training_data_filename)

    snd_layer_training_data_target = load_data_filter_id(
        TRAINING_DATA_FP, snd_layer_training_data.index, ['id', TARGET_LABEL])

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
        model_c_filepath = dirname + '/' + MODEL_CONSTITUTION_FILENAME
        cl_dict = load_json(model_c_filepath)
        cl_dict['snd_layer'] = dict()
        cl_dict['snd_layer']['models'] = model_dict_sub_l

        with open(model_c_filepath, 'w') as fp:
            json.dump(cl_dict, fp, indent=4)


def generate_models(strat, layer):

    dirname = ERA_CL_DIRNAME if strat == "cluster" else ERA_GRAPH_DIRNAME
    cl_dir_prefix = "cluster_"

    _, dirs, _ = next(os.walk(dirname))
    cl_dirname_l = [cl_dir for cl_dir in dirs if cl_dir_prefix in cl_dir]
    # cl_dirname_l = [cl_dirname_l[0]]  # DEBUG
    print("cl_dirname_l: ", cl_dirname_l)

    bDebug = False
    bMetrics = True
    bSaveModel = True

    bMultiProc = False
    bSaveModelDict = True

    if layer == 'fst':
        generate_fst_layer_model(dirname, cl_dirname_l, bDebug, bMetrics, bSaveModel,
                                 bMultiProc, bSaveModelDict)
    elif layer == 'snd':
        generate_snd_layer_model(dirname, bDebug, bMetrics, bSaveModel)

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


def load_valid_cl_data(data_fp, data_types, cols):
    f_r = ReaderCSV(data_fp)
    input_data = f_r.read_csv_matching('data_type', data_types, columns=cols)

    # TODO : move reindex to model_handler
    cols.remove('id')
    input_data = input_data.reindex(columns=cols)

    return input_data


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def make_fst_layer_model_params(eModel, model_prefix=None):
    if eModel == ModelType.RandomForest:
        n_est = [220]  # np.linspace(start=180, stop=250, num=20)
        max_d = [10]  # np.linspace(10, 20, num=5)
        model_params_array = map(
            lambda x: {
                'n_estimators': int(x[0]),
                'max_depth': int(x[1])
            }, itertools.product(*[n_est, max_d]))

        return model_params_array

    if eModel == ModelType.XGBoost:
        n_est = [220]  # np.linspace(start=180, stop=250, num=10)
        max_d = [10]  # np.linspace(10, 20, num=5)
        # eta = learning_rate
        eta = [1.0]  # np.logspace(start=(-1.0), stop=0.0, base=10.0, num=5)

        model_params_array = map(
            lambda x: {
                'n_estimators': int(x[0]),
                'max_depth': int(x[1]),
                'learning_rate': float(x[2])
            }, itertools.product(*[n_est, max_d, eta]))

        return model_params_array

    if eModel == ModelType.NeuralNetwork:
        num_layers = [1]  # np.linspace(start=1, stop=4, num=1)
        layer_size_factor = [0.66]
        train_batch_size = [50]
        num_epoch = [30]
        model_params_array = map(
            lambda x: {
                'num_layers': int(x[0]),
                'size_factor': float(x[1]),
                'train_batch_size': int(x[2]),
                'num_epoch': int(x[3])
            },
            itertools.product(
                *[num_layers, layer_size_factor, train_batch_size, num_epoch]))

        return model_params_array

    if eModel == ModelType.K_NN:
        leaf_size = [20]  # np.linspace(start=20, stop=50, num=1)
        minkowski_dist = [1]  # [1, 2, 3, 4]
        model_params_array = map(
            lambda x: {
                'n_neighbors': int(model_prefix),
                'leaf_size': int(x[0]),
                'minkowski_dist': int(x[1])
            }, itertools.product(*[leaf_size, minkowski_dist]))

        return model_params_array


def make_snd_layer_model_params(eModel, model_prefix=None):
    if eModel == ModelType.RandomForest:
        n_est = [150]  # np.linspace(start=180, stop=250, num=7)
        max_d = [10]  # np.linspace(10, 20, num=5)
        model_params_array = map(
            lambda x: {
                'n_estimators': int(x[0]),
                'max_depth': int(x[1])
            }, itertools.product(*[n_est, max_d]))

        return model_params_array

    if eModel == ModelType.XGBoost:
        n_est = [150]  # np.linspace(start=180, stop=250, num=7)
        max_d = [5]  # np.linspace(5, 20, num=5)
        # eta = learning_rate
        eta = [1.0]  # np.logspace(start=(-1.0), stop=0.0, base=10.0, num=5)

        model_params_array = map(
            lambda x: {
                'n_estimators': int(x[0]),
                'max_depth': int(x[1]),
                'learning_rate': float(x[2])
            }, itertools.product(*[n_est, max_d, eta]))

        return model_params_array

    if eModel == ModelType.NeuralNetwork:
        num_layers = [1]  # np.linspace(start=1, stop=4, num=1)
        layer_size_factor = [0.66]  # [0.33, 0.5, 0.66]
        train_batch_size = [50000]
        num_epoch = [25]
        model_params_array = map(
            lambda x: {
                'num_layers': int(x[0]),
                'size_factor': float(x[1]),
                'train_batch_size': int(x[2]),
                'num_epoch': int(x[3])
            },
            itertools.product(
                *[num_layers, layer_size_factor, train_batch_size, num_epoch]))

        return model_params_array

    if eModel == ModelType.K_NN:
        leaf_size = [20]  # np.linspace(start=20, stop=50, num=1)
        minkowski_dist = [1]  # [1, 2, 3, 4]
        model_params_array = map(
            lambda x: {
                'n_neighbors': int(model_prefix),
                'leaf_size': int(x[0]),
                'minkowski_dist': int(x[1])
            }, itertools.product(*[leaf_size, minkowski_dist]))

        return model_params_array


def make_model_prefix(eModel):
    if eModel is not ModelType.K_NN:
        return [None]

    return [5, 10, 30]


def gen_aggr_dict(cl_dict):
    cl_m_w_sorted = []
    for cl, cl_desc in cl_dict.items():
        cl_desc_m = cl_desc['models']
        for model_name, model_desc in cl_desc_m.items():
            cl_m_w = model_desc['valid_eval']['valid_score']['valid_corr_mean']
            if cl_m_w > 0:
                cl_m_w_sorted.append((cl, model_name, cl_m_w))

    cl_m_w_sorted = sorted(cl_m_w_sorted, key=lambda x: x[2], reverse=True)

    aggr_dict = dict()
    for i in range(1, len(cl_m_w_sorted)):
        aggr_cl_m_w = cl_m_w_sorted[:i]
        aggr_dict[i] = {
            'cluster_models': [(cl, m, w) for cl, m, w in aggr_cl_m_w],
            'total_w': sum([w for *_, w in aggr_cl_m_w])
        }

    return aggr_dict


def cl_model_build(dirname, cl, cl_dict, bMetrics=False, model_debug=False):
    print("build model cluster: ", cl)

    cl_dirpath = dirname + '/' + cl
    train_filepath = cl_dirpath + '/' + 'training_data.csv'
    # test_filepath = cl_dirpath + '/' + 'test_data.csv'

    train_data = load_data(train_filepath)
    #test_data = load_data(test_filepath)

    cl_fts = cl_dict['selected_features']

    valid_col = ['id', 'era'] + cl_fts + ['target']
    cl_valid_data = load_valid_cl_data(TOURNAMENT_DATA_FP, [VALID_TYPE],
                                       valid_col)

    # bMultiProc = False
    # if bMultiProc:
    # Parallelism only shows a slight speed improvment
    # -> not enough memory for mutliple thread w/ full dataset (?)
    # model_metrics_arg = list(zip(itertools.repeat(cl), itertools.repeat(
    #     metrics_filename), itertools.repeat(train_data), itertools.repeat(test_data), model_params_array))
    # pool_map(model_itf.generate_model, model_metrics_arg)
    # else:

    # model_types = ModelType
    # model_types = [ModelType.NeuralNetwork]
    # model_types = [ModelType.XGBoost, ModelType.RandomForest,
    #                ModelType.NeuralNetwork]  # , ModelType.K_NN]
    model_types = [
        ModelType.XGBoost, ModelType.RandomForest, ModelType.NeuralNetwork
    ]

    model_generator = ModelGenerator(cl_dirpath)
    train_input, train_target = model_generator.format_train_data(train_data)

    model_l = dict()
    for model_type in model_types:
        for model_prefix in make_model_prefix(model_type):

            model_generator.start_model_type(model_type, model_prefix,
                                             bMetrics)

            model_params_array = make_fst_layer_model_params(
                model_type, model_prefix)

            best_ll = sys.float_info.max
            for model_params in model_params_array:

                model_dict = dict()

                model_generator.generate_model(model_params, model_debug)
                model = model_generator.build_model(train_input, train_target)
                # print(" === evaluation - test data ===")
                # test_eval = model_generator.evaluate_model(test_data)
                print(" === tourn. validation - test data ===")
                valid_eval = model_generator.evaluate_model(
                    cl_valid_data, cl_dirpath)

                log_loss = valid_eval['log_loss']

                if log_loss < best_ll:
                    best_ll = log_loss
                    filepath, configpath = model.save_model()
                    # model_dict['test_eval'] = test_eval
                    model_dict['valid_eval'] = valid_eval
                    model_dict['model_filepath'] = filepath
                    model_dict['config_filepath'] = configpath
                    model_l[model_type.name] = model_dict

    cl_dict['models'] = model_l


def generate_cl_model(dirname, cl):

    bDebug = True
    bMetrics = True
    bSaveModel = False

    model_filepath = dirname + '/' + MODEL_CONSTITUTION_FILENAME
    model_dict = load_json(model_filepath)
    cl_dict = model_dict['clusters'][cl]

    cl_model_build(dirname, cl, cl_dict, bMetrics, bDebug)

    if bSaveModel:
        print("model_c_filepath: ", model_filepath)

        with open(model_filepath, 'w') as fp:
            json.dump(model_dict, fp, indent=4)


def generate_fst_layer_model(dirname, bDebug, bMetrics, bSaveModel, bMultiProc,
                             bSaveModelDict):
    model_c_filepath = dirname + '/' + MODEL_CONSTITUTION_FILENAME
    model_a_filepath = dirname + '/' + MODEL_AGGREGATION_FILENAME
    model_dict = load_json(model_c_filepath)
    cl_dict = model_dict['clusters']

    start_time = time.time()

    # Seems there is a pb with multiprocess (mult. proc w/ same dataframe?)
    model_dict_l = {}
    if bMultiProc:
        models_build_arg = list(
            zip(itertools.repeat(dirname), cl_dict.items(),
                itertools.repeat(bMetrics), itertools.repeat(bDebug)))
        cl_dir_model_l = pool_map(cl_model_build,
                                  models_build_arg,
                                  collect=True,
                                  arg_tuple=True)

        model_dict_l = dict(cl_dir_model_l)
    else:
        for cl, cl_dict in cl_dict.items():
            cl_model_build(dirname, cl, cl_dict, bMetrics, bDebug)
            continue

    print("model building done")
    print("--- %s seconds ---" % (time.time() - start_time))

    aggr_dict = gen_aggr_dict(model_dict['clusters'])

    if bSaveModel or bSaveModelDict:
        print("model_c_filepath: ", model_c_filepath)
        print("model_dict: ", model_dict)
        print("model_dict_l.keys(): ", model_dict_l.keys())

        with open(model_c_filepath, 'w') as fp:
            json.dump(model_dict, fp, indent=4)

        with open(model_a_filepath, 'w') as fp:
            json.dump(aggr_dict, fp, indent=4)


def load_data_filter_id(data_filename, list_id, columns=None):

    print("list_id: ", list_id)
    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv_filter_id(list_id,
                                             columns=columns).set_index("id")

    return data_df


def snd_layer_model_build(cl_dict,
                          snd_layer_dirname,
                          full_train_data,
                          bSaveModel=False,
                          bMetrics=False,
                          model_debug=False):

    input_models = ['XGBoost', 'RandomForest', 'NeuralNetwork']

    f_train_data = pd.DataFrame()
    for cl in cl_dict['clusters']:
        aux_df = full_train_data.loc[:,
                                     full_train_data.columns.str.
                                     startswith(cl + '_')]
        for i_m in input_models:
            i_m_train_data = aux_df.loc[:, aux_df.columns.str.find(i_m) > -1]
            f_train_data = pd.concat([f_train_data, i_m_train_data], axis=1)
    f_train_target_data = pd.concat(
        [f_train_data, full_train_data[TARGET_LABEL]], axis=1)

    print("f_train_data: ", f_train_target_data)
    train_data, test_data = train_test_split(f_train_target_data,
                                             test_size=TEST_RATIO)

    # model_types = [
    #     ModelType.XGBoost, ModelType.RandomForest, ModelType.NeuralNetwork
    # ]  # , ModelType.K_NN]
    model_types = [ModelType.NeuralNetwork]

    model_generator = ModelGenerator(snd_layer_dirname)
    train_input, train_target = model_generator.format_train_data(train_data)

    model_l = dict()
    model_l['input_models'] = input_models
    model_l['input_columns'] = f_train_data.columns.tolist()
    model_l['gen_models'] = []

    for model_type in model_types:
        for model_prefix in make_model_prefix(model_type):

            model_generator.start_model_type(model_type, model_prefix,
                                             bMetrics)

            model_params_array = make_snd_layer_model_params(
                model_type, model_prefix)

            best_ll = sys.float_info.max
            for model_params in model_params_array:

                print("generate model")
                model_generator.generate_model(model_params, model_debug)
                model = model_generator.build_model(train_input, train_target)
                model_dict = model_generator.evaluate_model(test_data)
                print("model: ", model)
                print("model_dict: ", model_dict)

                log_loss = model_dict['log_loss']

                if bSaveModel and (log_loss < best_ll):
                    best_ll = log_loss
                    filepath, configpath = model.save_model()
                    model_dict['model_filepath'] = filepath
                    model_dict['config_filepath'] = configpath
                    model_l['gen_models'].append(model_dict)

    return model_l


def generate_snd_layer_model(dirname, bDebug, bMetrics, bSaveModel):
    snd_layer_dirname = dirname + '/' + SND_LAYER_DIRNAME

    snd_layer_train_data_fp = dirname + '/' + PRED_TRAIN_FILENAME
    snd_layer_train_data = load_data(snd_layer_train_data_fp)

    print("snd_layer_training_data.columns: ", snd_layer_train_data.columns)

    model_c_filepath = dirname + '/' + MODEL_CONSTITUTION_FILENAME
    cl_dict = load_json(model_c_filepath)

    model_dict_sub_l = snd_layer_model_build(cl_dict,
                                             snd_layer_dirname,
                                             snd_layer_train_data,
                                             bSaveModel=bSaveModel,
                                             bMetrics=bMetrics,
                                             model_debug=bDebug)

    if bSaveModel:
        cl_dict[SND_LAYER]['models'] = model_dict_sub_l
        print("cl_dict: ", cl_dict[SND_LAYER])

        with open(model_c_filepath, 'w') as fp:
            json.dump(cl_dict, fp, indent=4)


def generate_models(strat_dir, strat, layer):

    bDebug = True
    bMetrics = True
    bSaveModel = False

    bMultiProc = False
    bSaveModelDict = True

    if layer == 'fst':
        generate_fst_layer_model(strat_dir, bDebug, bMetrics, bSaveModel,
                                 bMultiProc, bSaveModelDict)
    elif layer == 'snd':
        generate_snd_layer_model(strat_dir, bDebug, bMetrics, bSaveModel)

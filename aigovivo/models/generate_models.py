import os
import sys
import json
import numpy as np
import pandas as pd
import itertools
import json
import time
from sklearn.model_selection import train_test_split

from ..common import *
from ..threadpool import pool_map
from ..reader import ReaderCSV
from ..strat import StratConstitution

from .model_abstract import Model, ModelType
from .model_generator import ModelGenerator
from .model_params import model_params


def load_data(data_fp, cols=None):
    file_reader = ReaderCSV(data_fp)
    data_df = file_reader.read_csv(columns=cols).set_index("id")

    return data_df


def load_data_by_eras(data_fp, eras, cols=None):
    file_reader = ReaderCSV(data_fp)
    data_df = file_reader.read_csv_matching('era', eras,
                                            columns=cols).set_index("id")

    return data_df


def load_valid_cl_data(data_fp, data_types, cols):
    f_r = ReaderCSV(data_fp)
    valid_data = f_r.read_csv_matching('data_type', data_types,
                                       columns=cols).set_index('id')

    # TODO : move reindex to model_handler (model_generator)
    if 'id' in cols:
        cols.remove('id')

    valid_data = valid_data.reindex(columns=cols)

    return valid_data


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def make_model_prefix(eModel):
    if eModel is not ModelType.K_NN:
        return [None]

    return [5, 10, 30]


def gen_aggr_dict(cl_dict):
    cl_m_w_sorted = []

    # TODO: make aggregation as a mix of best model for each clusters

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
        aggr_pred_name = AGGR_PREFIX + str(i)
        aggr_dict[aggr_pred_name] = {
            'cluster_models': [(cl, m, w) for cl, m, w in aggr_cl_m_w],
            'total_w': sum([w for *_, w in aggr_cl_m_w])
        }

    return aggr_dict


def cl_model_build(dirname, cl, cl_dict, bMetrics=False, model_debug=False):
    print("build model cluster: ", cl)

    cl_dirpath = dirname + '/' + cl
    cl_eras = cl_dict['eras_name']

    cl_fts = cl_dict['selected_features']
    cl_cols = ['id', 'era'] + cl_fts + ['target']
    train_data = load_data_by_eras(TRAINING_DATA_FP, cl_eras, cols=cl_cols)

    # Need to reorder ft col by selected_features in model description
    cl_order_col = ['era'] + cl_fts + ['target']
    train_data = train_data[cl_order_col]

    cl_valid_data = load_valid_cl_data(TOURNAMENT_DATA_FP, [VALID_TYPE],
                                       cl_cols)

    # model_types = ModelType
    model_types = [ModelType.NeuralNetwork]
    # model_types = [ModelType.RandomForest]
    # model_types = [ModelType.XGBoost, ModelType.RandomForest,
    #                ModelType.NeuralNetwork]  # , ModelType.K_NN]
    # model_types = [
    #     ModelType.XGBoost, ModelType.RandomForest, ModelType.NeuralNetwork
    # ]

    model_generator = ModelGenerator(cl_dirpath)
    train_input, train_target = model_generator.format_train_data(train_data)

    for model_type in model_types:
        for model_prefix in make_model_prefix(model_type):

            model_generator.start_model_type(model_type, model_prefix,
                                             bMetrics)

            model_params_array = model_params('fst', model_type, model_prefix)

            best_ll = sys.float_info.max
            for m_params in model_params_array:

                model_dict = dict()

                model_generator.generate_model(m_params, model_debug)
                model = model_generator.build_model(train_input, train_target)
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
                    cl_dict['models'][model_type.name] = model_dict


def generate_cl_model(dirname, cl, bDebug, bMetrics, bSaveModel):

    strat_c_fp = dirname + '/' + STRAT_CONSTITUTION_FILENAME
    strat_c = StratConstitution(strat_c_fp)
    strat_c.load()

    cl_dict = strat_c.clusters[cl]

    cl_model_build(dirname, cl, cl_dict, bMetrics, bDebug)

    if bSaveModel:
        print("strat_c_fp: ", strat_c_fp)
        strat_c.save()


def generate_fst_layer_model(dirname, bDebug, bMetrics, bSaveModel, bMultiProc,
                             bSaveModelDict):
    strat_c_fp = dirname + '/' + STRAT_CONSTITUTION_FILENAME
    model_a_filepath = dirname + '/' + MODEL_AGGREGATION_FILENAME

    strat_c = StratConstitution(strat_c_fp)
    strat_c.load()

    cl_dict = strat_c.clusters

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
        for cl, cl_c in cl_dict.items():
            cl_model_build(dirname, cl, cl_c, bMetrics, bDebug)
            continue

    print("model building done")
    print("--- %s seconds ---" % (time.time() - start_time))

    if bSaveModel or bSaveModelDict:
        print("strat_c_fp: ", strat_c_fp)

        strat_c.save()

        aggr_dict = gen_aggr_dict(cl_dict)

        with open(model_a_filepath, 'w') as fp:
            json.dump(aggr_dict, fp, indent=4)


def load_data_filter_id(data_filename, list_id, columns=None):

    print("list_id: ", list_id)
    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv_filter_id(list_id,
                                             columns=columns).set_index("id")

    return data_df


def snd_layer_model_build(aggr_dict,
                          snd_layer_dirpath,
                          train_data_fp,
                          valid_data_fp,
                          bSaveModel=False,
                          bMetrics=False,
                          model_debug=False):

    model_aggr_dict = dict()

    for aggr_id, aggr_const in aggr_dict.items():

        input_cols = [
            cl + '_' + m for cl, m, _ in aggr_const['cluster_models']
        ]
        data_cols = ['id', 'era', TARGET_LABEL] + input_cols
        f_train_target_data = load_data(train_data_fp, cols=data_cols)
        f_valid_target_data = load_data(valid_data_fp, cols=data_cols)

        train_data, test_data = train_test_split(f_train_target_data,
                                                 test_size=TEST_RATIO)
        test_data['era'] = train_data['era']

        model_types = [
            ModelType.XGBoost, ModelType.RandomForest, ModelType.NeuralNetwork
        ]  # , ModelType.K_NN]
        #model_types = [ModelType.NeuralNetwork]

        model_generator = ModelGenerator(snd_layer_dirpath)
        train_input, train_target = model_generator.format_train_data(
            train_data)

        model_aggr_dict[aggr_id] = dict()
        current_aggr_dict = model_aggr_dict[aggr_id]
        current_aggr_dict['input_columns'] = input_cols
        current_aggr_dict['gen_models'] = dict()

        for model_type in model_types:
            for model_prefix in make_model_prefix(model_type):

                model_generator.start_model_type(model_type, model_prefix,
                                                 bMetrics)

                model_params_array = model_params('snd', model_type,
                                                  model_prefix)

                best_ll = sys.float_info.max
                for m_params in model_params_array:

                    print("generate model")
                    model_generator.generate_model(m_params, model_debug)
                    model = model_generator.build_model(
                        train_input, train_target)
                    model_dict = model_generator.evaluate_model(
                        f_valid_target_data, snd_layer_dirpath)
                    print("model: ", model)
                    print("model_dict: ", model_dict)

                    log_loss = model_dict['log_loss']

                    if bSaveModel and (log_loss < best_ll):
                        best_ll = log_loss
                        filepath, configpath = model.save_model()
                        model_dict['model_filepath'] = filepath
                        model_dict['config_filepath'] = configpath
                        current_aggr_dict['gen_models'][
                            model_type.name] = model_dict

    return model_aggr_dict


def generate_snd_layer_model(dirname, bDebug, bMetrics, bSaveModel):
    snd_layer_dirpath = dirname + '/' + SND_LAYER_DIRNAME

    snd_layer_train_data_fp = dirname + '/' + SND_LAYER_DIRNAME + '/' + PRED_TRAIN_FILENAME

    snd_layer_valid_data_fp = dirname + '/' + SND_LAYER_DIRNAME + '/' + PREDICTIONS_FILENAME + VALID_TYPE + PRED_FST_SUFFIX

    #print("snd_layer_training_data.columns: ", snd_layer_train_data.columns)
    #print("snd_layer_valid_data.columns: ", snd_layer_valid_data.columns)

    strat_c_fp = dirname + '/' + STRAT_CONSTITUTION_FILENAME
    strat_c = StratConstitution(strat_c_fp)
    strat_c.load()

    agg_filepath = dirname + '/' + MODEL_AGGREGATION_FILENAME
    aggr_dict = load_json(agg_filepath)

    model_aggr = snd_layer_model_build(aggr_dict,
                                       snd_layer_dirpath,
                                       snd_layer_train_data_fp,
                                       snd_layer_valid_data_fp,
                                       bSaveModel=bSaveModel,
                                       bMetrics=bMetrics,
                                       model_debug=bDebug)

    if bSaveModel:
        strat_c.snd_layer['models'] = model_aggr
        print("strat_c.snd_layer: ", strat_c.snd_layer)

        strat_c.save()


def generate_models(strat_dir, layer, bDebug, bMetrics, bSaveModel,
                    bMultiProc):

    bSaveModelDict = bSaveModel

    if layer == 'fst':
        generate_fst_layer_model(strat_dir, bDebug, bMetrics, bSaveModel,
                                 bMultiProc, bSaveModelDict)
    elif layer == 'snd':
        generate_snd_layer_model(strat_dir, bDebug, bMetrics, bSaveModel)

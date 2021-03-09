import os
import sys
import json
import random
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
from ..data_analysis import select_imp_ft
from ..models import Model, ModelType, ModelGenerator, model_params


def load_data(data_fp, cols=None):
    file_reader = ReaderCSV(data_fp)

    if cols is None:
        data_df = file_reader.read_csv().set_index("id")
    else:
        cols = ['id'] + cols
        data_df = file_reader.read_csv(columns=cols).set_index("id")

    if 'data_type' in data_df.columns:
        data_df = data_df.drop(['data_type'], axis=1)

    return data_df


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
            cl_m_w = model_desc['train_eval']['train_score']['corr_mean']
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


def build_evaluate_model(train_input,
                         train_target,
                         full_train_data,
                         model_gen,
                         cl_cols,
                         r_s,
                         b_p,
                         new_fts=None):
    if not b_p and new_fts is None:
        model = model_gen.build_model(train_input,
                                      train_target,
                                      random_search=r_s)

        print(" === evaluate model ===")
        print("full_train_data: ", full_train_data)
        train_eval = model_gen.evaluate_model(cl_cols, full_train_data)
    else:
        print("     -> use important features")
        print("     -> new_fts: ", new_fts)
        new_cols = ['era'] + new_fts
        train_ft_filter = train_input[new_cols]

        model = model_gen.build_model(train_ft_filter,
                                      train_target,
                                      random_search=r_s)

        print(" === evaluate model ===")
        new_cols = ['era'] + new_fts + ['target']
        train_eval = model_gen.evaluate_model(new_cols, full_train_data)

    print("train_eval: ", train_eval)
    return model, train_eval


def cl_model_build(dirname,
                   cl,
                   cl_dict,
                   model_types,
                   r_s,
                   b_p,
                   bFtImp=False,
                   bSaveModel=False,
                   bMetrics=False,
                   model_debug=False):
    print("build model cluster: ", cl)
    print("r_s: ", r_s)

    cl_dirpath = dirname + '/' + cl
    cl_eras = cl_dict['eras_name']

    # TODO : case when no selected_features
    # full_train_data = load_data(TRAINING_DATA_FP)
    # cl_fts = [x for x in full_train_data.columns if x.startswith('feature_')]
    # cl_cols = ['era'] + cl_fts + ['target']

    cl_fts = cl_dict['selected_features'].split('|')
    cl_cols = ['era'] + cl_fts + ['target']
    full_train_data = load_data(TRAINING_DATA_FP, cols=cl_cols)

    # Need to reorder ft col by selected_features in model description
    full_train_data = full_train_data[cl_cols]

    train_data = full_train_data.loc[full_train_data['era'].isin(cl_eras)]

    print('model_types: ', model_types)

    model_gen = ModelGenerator(cl_dirpath)
    train_input, train_target = model_gen.format_train_data(train_data)

    for model_type in model_types:

        orig_m_d = cl_dict['models'][
            model_type.name] if model_type.name in cl_dict['models'].keys(
            ) else None

        if orig_m_d is None and b_p:
            print('no original model to get best_params from')
            continue

        if 'best_params' not in orig_m_d.keys() and b_p:
            print('no best_params in original model')
            continue

        # K_NN not used for now
        #for model_prefix in make_model_prefix(model_type):
        model_prefix = None
        model_gen.start_model_type(model_type, model_prefix, bMetrics)

        model_params_array = [orig_m_d['best_params']
                              ] if b_p else model_params(
                                  'fst', model_type, model_prefix)

        best_ll = sys.float_info.max
        for m_params in model_params_array:

            model_dict = dict()

            new_fts = None
            if 'imp_fts' in orig_m_d.keys(
            ) and orig_m_d['imp_fts'] is not None:
                print("orig_m_d['imp_fts']: ", orig_m_d['imp_fts'])
                new_fts = orig_m_d['imp_fts'].split('|')

            model_gen.generate_model(m_params, model_debug)

            print(" === build model ===")
            model, train_eval = build_evaluate_model(train_input,
                                                     train_target,
                                                     full_train_data,
                                                     model_gen,
                                                     cl_cols,
                                                     r_s,
                                                     b_p,
                                                     new_fts=new_fts)

            print(" === features importance selection ===")

            if bFtImp:
                ft_sel = cl_fts if not b_p or new_fts is None else new_fts
                new_fts = select_imp_ft(full_train_data, model_type, ft_sel,
                                        model, train_eval['train_score'])

                print("     --> new_fts: ", new_fts)

                print(" === snd build model ===")
                model_2, train_eval_2 = build_evaluate_model(train_input,
                                                             train_target,
                                                             full_train_data,
                                                             model_gen,
                                                             cl_cols,
                                                             r_s,
                                                             b_p,
                                                             new_fts=new_fts)

                if train_eval_2['log_loss'] < train_eval['log_loss']:
                    model = model_2

            log_loss = train_eval['log_loss']
            if log_loss < best_ll and bSaveModel:
                best_ll = log_loss
                filepath, configpath = model.save_model()
                # model_dict['test_eval'] = test_eval
                model_dict[
                    'train_eval'] = train_eval_2 if bFtImp else train_eval
                if bFtImp:
                    model_dict['train_eval_cl_ft'] = train_eval
                model_dict['model_filepath'] = filepath
                model_dict['config_filepath'] = configpath
                model_dict['imp_fts'] = '|'.join(new_fts) if bFtImp else None
                model_dict['params'] = model.model_params
                if r_s:
                    model_dict['best_params'] = model.model_params
                model_dict['random_search'] = r_s
                cl_dict['models'][model_type.name] = model_dict


def generate_cl_model(dirname, cl, models, r_s, b_p, bFtImp, bDebug, bMetrics,
                      bSaveModel):

    strat_c_fp = dirname + '/' + STRAT_CONSTITUTION_FILENAME
    strat_c = StratConstitution(strat_c_fp)
    strat_c.load()

    cl_dict = strat_c.clusters[cl]

    cl_model_build(dirname, cl, cl_dict, models, r_s, b_p, bFtImp, bSaveModel,
                   bMetrics, bDebug)

    if bSaveModel:
        print("strat_c_fp: ", strat_c_fp)
        strat_c.save()


def generate_fst_layer_model(dirname, models, r_s, b_p, bFtImp, bDebug,
                             bMetrics, bSaveModel, bMultiProc):
    strat_c_fp = dirname + '/' + STRAT_CONSTITUTION_FILENAME
    model_a_filepath = dirname + '/' + MODEL_AGGREGATION_FILENAME

    strat_c = StratConstitution(strat_c_fp)
    strat_c.load()

    cl_dict = strat_c.clusters

    start_time = time.time()

    # Seems there is a pb with multiprocess (mult. proc w/ same dataframe?)
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
            cl_model_build(dirname, cl, cl_c, models, r_s, b_p, bFtImp,
                           bSaveModel, bMetrics, bDebug)
            continue

    print("model building done")
    print("--- %s seconds ---" % (time.time() - start_time))

    if bSaveModel:
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
                          model_types,
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
                        data_cols, f_valid_target_data)
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


def generate_snd_layer_model(dirname, model_types, bDebug, bMetrics,
                             bSaveModel):
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
                                       model_types,
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
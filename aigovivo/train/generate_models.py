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
from ..reader import ReaderCSV, load_h5_eras
from ..strat import StratConstitution, Aggregations
from ..data_analysis import select_imp_ft
from ..models import Model, ModelType, model_params, ModelDescription
from ..utils import get_eras

from .model_generator import ModelGenerator


def load_data(data_fp, eras, cols=None):

    data_df = load_h5_eras(data_fp, eras, cols=cols)

    # file_reader = ReaderCSV(data_fp)

    # if cols is None:
    #     data_df = file_reader.read_csv().set_index("id")
    # else:
    #     cols = ['id'] + cols
    #     data_df = file_reader.read_csv(columns=cols).set_index("id")

    if 'data_type' in data_df.columns:
        data_df = data_df.drop(['data_type'], axis=1)

    return data_df


def make_model_prefix(eModel):
    if eModel is not ModelType.K_NN:
        return [None]

    return [5, 10, 30]


def build_eval_model(train_input,
                     train_target,
                     full_t_data,
                     valid_data,
                     model_gen,
                     cl_cols,
                     r_s,
                     imp_fts=None):
    if imp_fts is None:
        model = model_gen.build_model(train_input,
                                      train_target,
                                      random_search=r_s)

        print(" === train evaluate model ===")
        train_eval = model_gen.evaluate_model(cl_cols, full_t_data)
        print(" === valid evaluate model ===")
        valid_eval = model_gen.evaluate_model(cl_cols, valid_data)
    else:
        print("     -> use important features")
        print("     -> imp_fts: ", imp_fts)
        new_cols = ['era'] + imp_fts
        train_ft_filter = train_input[new_cols]

        model = model_gen.build_model(train_ft_filter,
                                      train_target,
                                      random_search=r_s)

        print(" === train evaluate model ===")
        new_cols = ['era'] + imp_fts + ['target']
        train_eval = model_gen.evaluate_model(new_cols, full_t_data)
        print(" === valid evaluate model ===")
        valid_eval = model_gen.evaluate_model(new_cols, valid_data)

    print("train_eval: ", train_eval)
    return model, train_eval, valid_eval


def cl_model_random_search():
    return


def cl_model_standard_build(cl_dirpath,
                            cl_dict,
                            model_types,
                            bSaveModel=False,
                            bDebug=False):
    print("build model cluster: ", cl_dirpath)

    cl_eras = cl_dict['eras_name']

    # TODO : case when training on all features ?
    #     cl_fts = cl_dict['selected_features'].split('|')

    cl_fts = cl_dict['selected_features'].split('|')
    cl_cols = ['era'] + cl_fts + ['target']
    train_eras = get_eras('training')
    full_t_data = load_data(TRAINING_STORE_H5_FP, train_eras, cols=cl_cols)

    valid_eras = get_eras('validation')
    valid_data = load_data(TOURNAMENT_STORE_H5_FP, valid_eras, cols=cl_cols)

    # Need to reorder ft col by selected_features in cluster description
    full_t_data = full_t_data[cl_cols]
    train_data = full_t_data.loc[full_t_data['era'].isin(cl_eras)]

    model_gen = ModelGenerator(cl_dirpath)
    train_input, train_target = model_gen.format_train_data(train_data)

    cl_m = cl_dict['models']

    for eModel in model_types:
        model_desc_fp = cl_m[eModel.name]
        model_desc = ModelDescription('fst', model_desc_fp, eModel)
        model_desc.load()

        model_gen.start_model_type(eModel, None)

        m_params = model_desc.train_params

        print("model_params: ", m_params)

        m_params['num_eras'] = len(cl_eras)  # keep num eras info for model

        model_gen.generate_model(m_params, bDebug)

        model, t_eval, v_eval = build_eval_model(train_input,
                                                 train_target,
                                                 full_t_data,
                                                 valid_data,
                                                 model_gen,
                                                 cl_cols,
                                                 False,
                                                 imp_fts=None)

        if bSaveModel:
            m_fp, config_fp = model.save_model()

            model_desc.train_eval = t_eval
            model_desc.valid_eval = v_eval

            model_desc.model_fp = m_fp
            model_desc.config_fp = config_fp
            model_desc.train_params = m_params
            model_desc.save()

    return


def cl_model_metrics_build(cl_dirpath, cl_dict, model_types, bDebug=False):
    print("build model cluster: ", cl_dirpath)

    cl_eras = cl_dict['eras_name']

    # TODO : case when training on all features ?
    #     cl_fts = cl_dict['selected_features'].split('|')

    cl_fts = cl_dict['selected_features'].split('|')
    cl_cols = ['era'] + cl_fts + ['target']
    train_eras = get_eras('training')
    full_t_data = load_data(TRAINING_STORE_H5_FP, train_eras, cols=cl_cols)

    valid_eras = get_eras('validation')
    valid_data = load_data(TOURNAMENT_STORE_H5_FP, valid_eras, cols=cl_cols)

    # Need to reorder ft col by selected_features in cluster description
    full_t_data = full_t_data[cl_cols]
    train_data = full_t_data.loc[full_t_data['era'].isin(cl_eras)]

    model_gen = ModelGenerator(cl_dirpath)
    train_input, train_target = model_gen.format_train_data(train_data)

    for eModel in model_types:

        model_gen.start_model_type(eModel)

        model_params_array = model_params('fst', eModel, True)
        for m_params in model_params_array:
            m_params['num_eras'] = len(cl_eras)  # keep num eras info for model
            print("model_params: ", m_params)

            model_gen.generate_model(m_params, bDebug)

            model, t_eval, v_eval = build_eval_model(train_input,
                                                     train_target,
                                                     full_t_data,
                                                     valid_data,
                                                     model_gen,
                                                     cl_cols,
                                                     False,
                                                     imp_fts=None)

            model_gen.append_metrics(v_eval)

    return


def cl_model_build(cl_dirpath,
                   cl_dict,
                   model_types,
                   r_s,
                   b_p,
                   bFtImp=False,
                   bSaveModel=False,
                   bMetrics=False,
                   model_debug=False):
    print("build model cluster: ", cl_dirpath)
    print("r_s: ", r_s)

    cl_eras = cl_dict['eras_name']

    # TODO : case when no selected_features
    # full_t_data = load_data(TRAINING_DATA_FP)
    # cl_fts = [x for x in full_t_data.columns if x.startswith('feature_')]
    # cl_cols = ['era'] + cl_fts + ['target']

    cl_fts = cl_dict['selected_features'].split('|')
    cl_cols = ['era'] + cl_fts + ['target']
    train_eras = get_eras('training')
    full_t_data = load_data(TRAINING_STORE_H5_FP, train_eras, cols=cl_cols)

    valid_eras = get_eras('validation')
    valid_data = load_data(TOURNAMENT_STORE_H5_FP, valid_eras, cols=cl_cols)

    # Need to reorder ft col by selected_features in model description
    full_t_data = full_t_data[cl_cols]

    train_data = full_t_data.loc[full_t_data['era'].isin(cl_eras)]

    print('model_types: ', model_types)

    model_gen = ModelGenerator(cl_dirpath)
    train_input, train_target = model_gen.format_train_data(train_data)

    for model_type in model_types:

        orig_m_d = cl_dict['models'][
            model_type.name] if model_type.name in cl_dict['models'].keys(
            ) else None
        no_m_d = orig_m_d is None

        if no_m_d and b_p:
            print('no original model to get best_params from')
            continue

        has_b_p = False if no_m_d else 'best_params' not in orig_m_d.keys()
        if b_p and not has_b_p:
            print('no best_params in original model')
            continue

        # K_NN not used for now
        #for model_prefix in make_model_prefix(model_type):
        model_prefix = None
        model_gen.start_model_type(model_type, model_prefix)

        model_params_array = [orig_m_d['best_params']
                              ] if b_p else model_params(
                                  'fst', model_type, bMetrics, model_prefix)

        best_ll = sys.float_info.max
        for m_params in model_params_array:
            m_params['num_eras'] = len(cl_eras)  # keep num eras info for model

            model_dict = dict()

            imp_fts = None
            has_imp_fts = not no_m_d and 'imp_fts' in orig_m_d.keys()
            if has_imp_fts:
                print("orig_m_d['imp_fts']: ", orig_m_d['imp_fts'])
                orig_imp_fts = orig_m_d['imp_fts']
                if orig_imp_fts is not None:
                    imp_fts = orig_imp_fts.split('|')

            model_gen.generate_model(m_params, model_debug)

            print(" === build model ===")
            model, train_eval, valid_eval = build_eval_model(train_input,
                                                             train_target,
                                                             full_t_data,
                                                             valid_data,
                                                             model_gen,
                                                             cl_cols,
                                                             r_s,
                                                             imp_fts=imp_fts)
            if bMetrics:
                model_gen.append_metrics(valid_eval)

            if bFtImp:
                print(" === features importance selection ===")
                ft_sel = cl_fts if not b_p or imp_fts is None else imp_fts
                imp_fts = select_imp_ft(full_t_data, model_type, ft_sel, model,
                                        train_eval['eval_score'])

                print("     --> imp_fts: ", imp_fts)

                print(" === snd build model ===")
                model_2, train_eval_2, valid_eval_2 = build_eval_model(
                    train_input,
                    train_target,
                    full_t_data,
                    valid_data,
                    model_gen,
                    cl_cols,
                    r_s,
                    imp_fts=imp_fts)

                if valid_eval_2['log_loss'] < valid_eval_2['log_loss']:
                    model = model_2

            log_loss = train_eval['log_loss']
            if log_loss < best_ll and bSaveModel:
                best_ll = log_loss
                filepath, configpath = model.save_model()
                # model_dict['test_eval'] = test_eval
                model_dict[
                    'train_eval'] = train_eval_2 if bFtImp else train_eval
                model_dict[
                    'valid_eval'] = valid_eval_2 if bFtImp else valid_eval
                if bFtImp:
                    model_dict['train_eval_cl_ft'] = train_eval
                model_dict['model_filepath'] = filepath
                model_dict['config_filepath'] = configpath
                model_dict['imp_fts'] = '|'.join(imp_fts) if bFtImp else None
                model_dict['params'] = model.model_params
                if r_s:
                    model_dict['best_params'] = model.model_params
                model_dict['random_search'] = r_s
                cl_dict['models'][model_type.name] = model_dict


def generate_cl_model(dirname, cl, models, r_s, bFtImp, bDebug, bMetrics,
                      bSaveModel):

    strat_c_fp = dirname + '/' + STRAT_CONSTITUTION_FILENAME
    strat_c = StratConstitution(strat_c_fp)
    strat_c.load()

    cl_dict = strat_c.clusters[cl]

    cl_dirpath = dirname + '/' + cl

    if bFtImp:
        return
    elif bMetrics:
        cl_model_metrics_build(cl_dirpath, cl_dict, models, bDebug=bDebug)
        return
    elif r_s:
        return
    else:
        cl_model_standard_build(cl_dirpath, cl_dict, models, bSaveModel,
                                bDebug)

    if bSaveModel:
        print("strat_c_fp: ", strat_c_fp)
        strat_c.save()


def generate_fst_layer_model(dirname, models, r_s, bFtImp, bDebug, bMetrics,
                             bSaveModel):
    strat_c_fp = dirname + '/' + STRAT_CONSTITUTION_FILENAME
    strat_c = StratConstitution(strat_c_fp)
    strat_c.load()

    strat_clusters = strat_c.clusters

    start_time = time.time()

    # Seems there is a pb with multiprocess (mult. proc w/ same dataframe?)
    # if bMultiProc:
    #     models_build_arg = list(
    #         zip(itertools.repeat(dirname), cl_dict.items(),
    #             itertools.repeat(bMetrics), itertools.repeat(bDebug)))
    #     cl_dir_model_l = pool_map(cl_model_build,
    #                               models_build_arg,
    #                               collect=True,
    #                               arg_tuple=True)
    #     model_dict_l = dict(cl_dir_model_l)
    # else:
    for cl, cl_dict in strat_clusters.items():

        cl_dirpath = dirname + '/' + cl

        if bFtImp:
            return
        elif bMetrics:
            cl_model_metrics_build(cl_dirpath, cl_dict, models, bDebug=bDebug)
        elif r_s:
            return
        else:
            cl_model_standard_build(cl_dirpath, cl_dict, models, bSaveModel,
                                    bDebug)
        continue

    print("model building done")
    print("--- %s seconds ---" % (time.time() - start_time))

    if bSaveModel:
        print("strat_c_fp: ", strat_c_fp)

        strat_c.save()

        # make_aggr_dict(dirname)


def snd_layer_model_build(aggr_dict,
                          model_types,
                          snd_layer_dirpath,
                          train_data_fp,
                          valid_data_fp,
                          bSaveModel=False,
                          bMetrics=False,
                          model_debug=False):

    model_aggr_dict = dict()

    train_eras = get_eras('training')
    valid_eras = get_eras('validation')

    for aggr_id, aggr_const in aggr_dict.items():

        input_cols = [
            cl + '_' + m for cl, m, _ in aggr_const['cluster_models']
        ]
        data_cols = ['id', 'era', TARGET_LABEL] + input_cols
        f_train_target_data = load_data(train_data_fp,
                                        train_eras,
                                        cols=data_cols)
        f_valid_target_data = load_data(valid_data_fp,
                                        valid_eras,
                                        cols=data_cols)

        # train_data, test_data = train_test_split(f_train_target_data,
        #                                         test_size=TEST_RATIO)
        # test_data['era'] = train_data['era']
        train_data = f_train_target_data

        model_generator = ModelGenerator(snd_layer_dirpath)
        train_input, train_target = model_generator.format_train_data(
            train_data)

        model_aggr_dict[aggr_id] = dict()
        current_aggr_dict = model_aggr_dict[aggr_id]
        current_aggr_dict['input_columns'] = input_cols
        current_aggr_dict['gen_models'] = dict()

        for model_type in model_types:
            for model_prefix in make_model_prefix(model_type):

                model_generator.start_model_type(model_type, model_prefix)

                model_params_array = model_params('snd', model_type, bMetrics,
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
    aggr_dict = Aggregations(agg_filepath)
    aggr_dict.load()

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

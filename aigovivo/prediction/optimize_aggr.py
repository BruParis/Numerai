import json
import math
import copy
import time
import pandas as pd
import numpy as np

from ..data_analysis import pred_score, rank_pred, neutralize
from ..reader import ReaderCSV, load_h5_eras
from ..utils import get_eras
from ..common import *

from .pred_diagnostics import ft_exp_analysis

K_MAX = 1000
SCORE_PROBA_THR = 0.1
K_PROBA_THR = 0.1
K_EXP_CONST = K_MAX / 3


def load_data(data_filepath, cols=None):
    file_reader = ReaderCSV(data_filepath)
    input_data = file_reader.read_csv(columns=cols).set_index('id')

    return input_data


def load_valid_data(eras):

    start_time = time.time()
    input_data = load_h5_eras(TOURNAMENT_STORE_H5_FP, eras)
    print("--- %s seconds ---" % (time.time() - start_time))

    return input_data


def load_valid_target():
    valid_eras = get_eras(VALID_TYPE)
    valid_cols = ['era', TARGET_LABEL]
    valid_target = load_h5_eras(TOURNAMENT_STORE_H5_FP,
                                valid_eras,
                                cols=valid_cols)

    return valid_target


def load_pred_aggr_orig(strat_dir, aggr_orig, cl_fn):
    print("aggr_l: ", aggr_orig)

    rank_l = []

    for cl, model_n, _ in aggr_orig['cluster_models']:
        cl_fp = strat_dir + '/' + cl + '/' + cl_fn
        cl_rank = load_data(cl_fp, cols=['id', model_n])
        rank_dict = {'cluster': cl, 'model': model_n, 'rank': cl_rank}
        rank_l.append(rank_dict)

    return rank_l


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def aggr_pred(rank_l, w_dict):

    full_pred_df = pd.DataFrame(columns=['Full'])
    total_w = 0
    for rank in rank_l:
        cl = rank['cluster']
        m_n = rank['model']
        cl_m_rank = rank['rank'].copy()
        cl_m_rank.columns = ['Full']

        weight = w_dict[cl][m_n]

        if full_pred_df.size == 0:
            full_pred_df = cl_m_rank * weight
        else:
            full_pred_df = full_pred_df + cl_m_rank * weight

        total_w += weight

    full_pred_df /= total_w
    full_rank_df = rank_pred(full_pred_df)

    return full_rank_df


def neighbour_weights(w_dict):
    w_l = []
    for _, cl_m_w in w_dict.items():
        for _, w in cl_m_w.items():
            w_l.append(w)
    std_w = np.std(w_l)

    w_n_dict = copy.deepcopy(w_dict)
    for cl, cl_m_w in w_n_dict.items():
        for m, w in cl_m_w.items():
            rand_num = np.random.uniform(-1, 1)
            delta = rand_num * std_w
            w_n_dict[cl][m] = w + delta

    return w_n_dict


def proba_keep_step(new_score, current_score, k):

    rel_diff = (current_score - new_score) / current_score
    if rel_diff > SCORE_PROBA_THR:
        return False

    k_ratio = (K_MAX - k) / K_EXP_CONST
    temp = math.exp(-k_ratio)
    rand_num = np.random.uniform(0, 1)

    return temp > rand_num


def optim_annealing(rank_orig_l, valid_target, score_orig, w_dict):

    k = K_MAX
    score = score_orig
    w_res_dict = w_dict.copy()
    while k > 0:
        w_new_dict = neighbour_weights(w_res_dict)

        new_aggr_pred_df = aggr_pred(rank_orig_l, w_new_dict)
        new_score = pred_score(new_aggr_pred_df,
                               'Full',
                               valid_target,
                               bDebug=False)['corr_mean']

        # if new_score > score or proba_keep_step(new_score, score, k):
        if new_score > score:
            score = new_score
            w_res_dict = w_new_dict
            print("k: {} - new score: {}".format(k, score))

        k -= 1

    return w_res_dict, score


def optimize_aggr(rank_orig_l, w_dict, valid_target, orig_key):

    aggr_pred_df = aggr_pred(rank_orig_l, w_dict)
    score_orig = pred_score(aggr_pred_df, 'Full', valid_target,
                            bDebug=False)['corr_mean']
    print("score_orig: ", score_orig)

    optim_w_dict, score = optim_annealing(rank_orig_l, valid_target,
                                          score_orig, w_dict)

    optim_w_l = []
    optim_total_w = 0
    for cl, cl_m_w in optim_w_dict.items():
        for m, w in cl_m_w.items():
            optim_w_l.append([cl, m, w])
            optim_total_w += w

    aggr_optim = {
        'cluster_models': optim_w_l,
        'total_w': optim_total_w,
        'orig_aggr': orig_key,
        'corr_mean': score
    }

    return aggr_optim


def optimize_neutr(model_aggr_fp, rank_orig_l, aggr_d, w_dict, valid_data,
                   valid_eras):

    fts = [ft for ft in valid_data.columns if ft.startswith('feature_')]

    aggr_pred_df = aggr_pred(rank_orig_l, w_dict)

    neutr_col_name = 'neutralized'
    valid_pred_df = pd.concat([valid_data, aggr_pred_df], axis=1)
    valid_pred_df[neutr_col_name] = valid_pred_df.groupby('era').apply(
        lambda x: neutralize(x, 'Full')).values
    valid_pred_df[neutr_col_name] = neutralize(valid_pred_df,
                                               'Full',
                                               proportion=1.0)
    print("valid_pred_df: ", valid_pred_df)

    diag_d = ft_exp_analysis(valid_pred_df, valid_eras, neutr_col_name)
    orig_corr = diag_d['mean_corr']
    print(" ---> original mean corr: ", orig_corr)

    rem_ft = []
    for ft in fts:
        print("ft: ", ft)
        fts_r = [ft_0 for ft_0 in fts if ft_0 != ft]
        valid_col = fts_r + ['era', TARGET_LABEL]
        valid_rem_ft_df = pd.concat([valid_data[valid_col], aggr_pred_df],
                                    axis=1)
        valid_rem_ft_df[neutr_col_name] = neutralize(valid_rem_ft_df,
                                                     'Full',
                                                     proportion=1.0)

        diag_ft_d = ft_exp_analysis(valid_rem_ft_df, valid_eras,
                                    neutr_col_name)
        ft_corr = diag_ft_d['mean_corr']

        if orig_corr < ft_corr:
            print("ft {} corr: {}".format(ft, ft_corr))
            rem_ft.append(ft)

    sel_ft = [ft for ft in fts if ft not in rem_ft]
    valid_sel_col = sel_ft + ['era', TARGET_LABEL]
    valid_sel_ft_df = valid_data[valid_sel_col]
    valid_sel_ft_df = pd.concat([valid_sel_ft_df, aggr_pred_df], axis=1)
    valid_sel_ft_df[neutr_col_name] = neutralize(valid_sel_ft_df,
                                                 'Full',
                                                 proportion=1.0)

    new_diag_d = ft_exp_analysis(valid_sel_ft_df, valid_eras, neutr_col_name)
    new_corr = new_diag_d['mean_corr']
    print("    ---> new corr: {}".format(new_corr))

    sel_ft_str = '|'.join(sel_ft)
    aggr_d['optim_neutr'] = {'sel_ft': sel_ft_str, 'corr_mean': new_corr}


def optim_proc(strat_dir, model_types, aggr_id, neutr=None):

    model_aggr_fp = strat_dir + '/' + MODEL_AGGREGATION_FILENAME

    aggr_key = AGGR_PREFIX + aggr_id
    models_key = ",".join([m.name for m in model_types])
    aggr_dict = load_json(model_aggr_fp)
    aggr_orig_dict = aggr_dict[models_key][aggr_key]

    proba_cl_fn = PROBA_FILENAME + VALID_TYPE + '.csv'

    rank_orig_l = load_pred_aggr_orig(strat_dir, aggr_orig_dict, proba_cl_fn)

    w_dict = dict()
    for cl, model_n, w in aggr_orig_dict['cluster_models']:
        if cl not in w_dict.keys():
            w_dict[cl] = dict()
        w_dict[cl][model_n] = w
    print('w_dict: ', w_dict)

    if neutr:
        valid_eras = get_eras(VALID_TYPE)
        print("valid_eras: ", valid_eras)
        valid_data = load_valid_data(valid_eras)

        optimize_neutr(model_aggr_fp, rank_orig_l, aggr_orig_dict, w_dict,
                       valid_data, valid_eras)
    else:
        valid_target = load_valid_target()
        aggr_optim = optimize_aggr(rank_orig_l, w_dict, valid_target, aggr_key)

        aggr_dict[models_key]['aggr_pred_optim'] = aggr_optim

    with open(model_aggr_fp, 'w') as fp:
        json.dump(aggr_dict, fp, indent=4)

    return
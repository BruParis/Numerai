import json
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..strat import StratConstitution
from ..common import *
from ..reader import load_h5_eras
from ..prediction import PredictionOperator, model_predic_rank, indiv_ft_exp, neutralize
from ..models import ModelType, load_model
from ..utils import get_eras

from .ranking import rank_proba_models, rank_pred
from .feature_era_corr import compute_corr
from .validation_corr import pred_score

K_REPETITIONS = 4


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def plot_ft_exp(ft_exp_df, model_ft):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

    print("ft_exp_df: ", ft_exp_df)
    i_ft_exp = ft_exp_df['exp']
    i_ft_exp_n = ft_exp_df['exp_n']

    color_b = [
        'green' if ft in model_ft else 'blue' for ft in ft_exp_df['ft'].values
    ]

    ax1.bar(ft_exp_df['ft'], i_ft_exp, color=color_b)
    ax1.axes.get_xaxis().set_visible(False)

    ax2.bar(ft_exp_df['ft'], i_ft_exp_n, color=color_b)
    ax2.axes.get_xaxis().set_visible(False)

    plt.show()

    return


def make_ft_exp(fts, cl_m_train_data, model_n, model_n_neutr):
    i_ft_exp = np.abs(indiv_ft_exp(cl_m_train_data, model_n))
    i_ft_exp_n = np.abs(indiv_ft_exp(cl_m_train_data, model_n_neutr))

    ft_exp_df = pd.DataFrame(data={
        'ft': fts,
        'exp': i_ft_exp,
        'exp_n': i_ft_exp_n
    })

    plot_ft_exp(ft_exp_df, fts)

    return


def select_imp_ft(train_data, eModel, m_ft, model, train_eval):
    ref_corr = train_eval['corr_mean']

    imp_d = dict()
    for ft in m_ft:
        print("ft: ", ft)
        ft_corr = 0
        corrupt_train_df = train_data.copy()
        for _ in range(K_REPETITIONS):
            # corrupt_train_df[ft] = corrupt_train_df.groupby(
            #     "era")[ft].transform(np.random.permutation)
            ft_era_groups = [df for _, df in train_data.groupby('era')[ft]]
            random.shuffle(ft_era_groups)
            corrupt_train_df[ft] = pd.concat(ft_era_groups, axis=0).values

            corrupt_rank = model_predic_rank(eModel, model, m_ft,
                                             corrupt_train_df)
            corrupt_rank['era'] = corrupt_train_df['era']
            # score_k = compute_corr(corrupt_rank,
            #                        cl_m_train_data[TARGET_LABEL])
            score_k = pred_score(corrupt_rank,
                                 eModel.name,
                                 train_data[TARGET_LABEL],
                                 bDebug=False)

            ft_corr_k = score_k['corr_mean']

            ft_corr += ft_corr_k

        ft_corr /= K_REPETITIONS

        if ft_corr > ref_corr:
            print(" -> ft_corr: ", ft_corr)

        imp_d[ft] = ref_corr - ft_corr

    sel_fts = [ft for ft, score in imp_d.items() if score > 0.0]

    return sel_fts


def model_ft_imp(strat_dir, cl, model_types, metrics=False, save=False):

    print("strat_dir: ", strat_dir)
    print("cl: ", cl)

    cl_path = strat_dir + '/' + cl
    strat_c_fp = strat_dir + '/' + STRAT_CONSTITUTION_FILENAME
    strat_c = StratConstitution(strat_c_fp)
    strat_c.load()

    valid_eras = get_eras(VALID_TYPE)

    valid_data = load_h5_eras(TOURNAMENT_STORE_H5_FP, valid_eras)

    print("valid_data: ", valid_data)

    for eModel in model_types:
        model_n = eModel.name

        cl_model_d = strat_c.clusters[cl]

        # cl_m_eras = cl_model_d['eras_name']
        cl_m_eras = get_eras(TRAINING_TYPE)
        cl_m_fts = cl_model_d['selected_features'].split('|')
        cl_m_desc = cl_model_d['models'][model_n]

        cl_m_prefix = None if 'prefix' not in cl_m_desc.keys(
        ) else cl_m_desc['prefix']
        model = load_model(cl_path, eModel, cl_m_prefix)

        # predict against train to select features
        # using validation data would lead to overfitting
        cl_m_train_data = load_h5_eras(TRAINING_STORE_H5_FP,
                                       cl_m_eras).set_index('id')
        # if no selected_ffeatures
        # cl_m_fts = [
        #     x for x in cl_m_train_data.columns if x.startswith('feature_')
        # ]

        cl_m_rank = model_predic_rank(eModel, model, cl_m_fts, cl_m_train_data)
        cl_m_rank['era'] = cl_m_train_data['era']

        # ref_score = compute_corr(cl_m_rank[model_n],
        #                          cl_m_train_data[TARGET_LABEL])
        ref_score = pred_score(cl_m_rank, model_n,
                               cl_m_train_data[TARGET_LABEL])
        ref_corr = ref_score['corr_mean']

        print("cl_m_rank: ", cl_m_rank)
        print("ref_score: ", ref_score)

        # TODO : features clustering by correlation? with k-clusters? with hierar.linkage?
        # If a group if feature is strongly correlated, model will still have acces
        # to premuted ft through its correlated ft.

        select_imp_ft(cl_m_train_data, eModel, cl_m_fts, model, ref_score)

    return


# cl_m_train_data = pd.concat([cl_m_train_data, cl_m_rank[model_n]],
#                             axis=1
# model_n_neutr = model_n + '_neutralized
# print("cl_m_train_data: ", cl_m_train_data)
# print("cl_m_train_data columns: ", cl_m_train_data.columns
# cl_m_train_data[model_n_neutr] = cl_m_train_data.groupby('era').apply(
#     lambda x, m_n=model_n: neutralize(x, m_n)).values
# cl_m_train_data[model_n_neutr] = rank_pred(
#     cl_m_train_data[model_n_neutr]
# make_ft_exp(cl_m_fts, cl_m_train_data, model_n, model_n_neutr
# K predictions for each ft
# with one-by-one era-wise shuffle of ft dat
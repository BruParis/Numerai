import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.table as table
from scipy.stats import spearmanr

from ..common import *
from ..strat import StratConstitution
from ..reader import ReaderCSV, load_h5_eras
from ..utils import get_eras
from ..data_analysis import compute_corr, neutralize, rank_pred


def load_valid_cl_ft_target(eras, cl_ft=None):

    cols = ['era'] + cl_ft + ['target'] if cl_ft is not None else None

    start_time = time.time()
    input_data = load_h5_eras(TOURNAMENT_STORE_H5_FP, eras, cols=cols)
    print("--- %s seconds ---" % (time.time() - start_time))

    return input_data


def load_data(data_filepath, cols=None):
    file_reader = ReaderCSV(data_filepath)
    input_data = file_reader.read_csv(columns=cols).set_index('id')

    return input_data


def plot_diag(valid_eras, indiv_ft_exp, ft_exp, max_ft_exp, cl_era_corr,
              max_drawdown, mean_corr, valid_sd, sharp, full_corr, pic_fp):
    fig, ((ax1, ax2), (ax1_t, ax2_t)) = plt.subplots(2,
                                                     2,
                                                     figsize=(20, 10),
                                                     gridspec_kw={
                                                         'width_ratios':
                                                         [2, 1],
                                                         'height_ratios':
                                                         [5, 1]
                                                     })

    ax1.bar(range(len(indiv_ft_exp)), indiv_ft_exp)
    ax1.axes.get_xaxis().set_visible(False)

    ax1_t.axis("off")
    celltext = [[f"{ft_exp:.4f}"], [f"{max_ft_exp:.4f}"]]
    table_1 = table.table(ax1_t,
                          cellText=celltext,
                          colWidths=[0.6, 0.4],
                          rowLabels=["ft exp", "max ft exp"],
                          loc='top left')
    table_1.scale(1, 4)

    ax2.bar(valid_eras, cl_era_corr)
    ax2.axhline(y=full_corr, color="red")
    ax2.text(0, full_corr, "{:.4f}".format(full_corr), color="red")

    ax2_t.axis("off")
    celltext = [[f"{full_corr:.4f}"], [f"{mean_corr:.4f}"],
                [f"{valid_sd:.4f}"], [f"{sharp:.4f}"], [f"{max_drawdown:.4f}"]]
    table_2 = table.table(ax2_t,
                          cellText=celltext,
                          colWidths=[0.6, 0.4],
                          rowLabels=[
                              "full corr", "mean corr", "valid_sd",
                              "sharpe ratio", "max drawdown"
                          ])
    table_2.scale(1, 2)

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=60)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=60)

    fig.savefig(pic_fp)


def indiv_ft_exp(df, rank_name):
    feature_names = [f for f in df.columns if f.startswith("feature")]
    exposures = []

    for f in feature_names:
        fe = spearmanr(df[rank_name], df[f])[0]
        exposures.append(fe)
    return np.array(exposures)


def max_ft_exp(df, rank_name):
    return np.max(np.abs(indiv_ft_exp(df, rank_name)))


def ft_exp(df, rank_name):
    return np.sqrt(np.mean(np.square(indiv_ft_exp(df, rank_name))))


def ft_exp_analysis(data_ft_rank, valid_eras, rank_col_name, pic_fp):

    target_data = data_ft_rank[['era', TARGET_LABEL]]

    ind_ft_exp = np.abs(indiv_ft_exp(data_ft_rank, rank_col_name))
    ft_e = ft_exp(data_ft_rank, rank_col_name)
    max_ft_e = max_ft_exp(data_ft_rank, rank_col_name)

    print("ft_e: ", ft_e)
    print("max_ft_e: ", max_ft_e)

    print("target_data: ", target_data)
    print("data_ft_rank: ", data_ft_rank)

    cl_era_corr = [
        compute_corr(
            target_data.loc[target_data['era'] == era][TARGET_LABEL],
            data_ft_rank.loc[data_ft_rank['era'] == era][rank_col_name])
        for era in valid_eras
    ]

    full_corr = compute_corr(target_data[TARGET_LABEL],
                             data_ft_rank[rank_col_name])

    cumul_max = (pd.Series(cl_era_corr) + 1).cumprod().rolling(
        window=100, min_periods=1).max()
    max_drawdown = (cumul_max - (pd.Series(cl_era_corr) + 1).cumprod()).max()

    mean_corr = np.mean(cl_era_corr)
    valid_sd = np.std(cl_era_corr)
    sharp = mean_corr / valid_sd

    print("full_corr: ", full_corr)
    print("mean_corr: ", mean_corr)
    print("valid_sd: ", valid_sd)
    print("sharp: ", sharp)
    print("max_drawdown: ", max_drawdown)
    plot_diag(valid_eras, ind_ft_exp, ft_e, max_ft_e, cl_era_corr,
              max_drawdown, mean_corr, valid_sd, sharp, full_corr, pic_fp)

    diag_dict = {
        'full_corr': full_corr,
        'mean_corr': mean_corr,
        'valid_sd': valid_sd,
        'sharp': sharp,
        'max_drawdown': max_drawdown,
        'ft_exp': ft_e,
        'max_ft_exp': max_ft_e
    }

    return diag_dict


def cl_pred_diagnostics(strat_dir, cluster):

    strat_c_fp = strat_dir + '/' + STRAT_CONSTITUTION_FILENAME
    strat_c = StratConstitution(strat_c_fp)
    strat_c.load()

    cl_dict = strat_c.clusters[cluster]

    cl_fts = cl_dict['selected_features']

    valid_eras = get_eras(VALID_TYPE)

    valid_data = load_valid_cl_ft_target(valid_eras, cl_fts)
    for model_name, cl_model_dict in cl_dict['models'].items():
        cl_m_valid_d = cl_model_dict['valid_eval']
        if 'valid_data_fp' not in cl_m_valid_d.keys():
            continue

        cl_rank_fp = cl_m_valid_d['valid_data_fp']
        cl_m_rank = load_data(cl_rank_fp, cols=['id', 'era', model_name])
        cl_m_rank = cl_m_rank[['era', model_name]]

        data_ft_rank = pd.concat([valid_data, cl_m_rank], axis=1)

        pic_fp = strat_dir + '/' + cluster + '/' + model_name + '_ft_exp.png'
        ft_exp_analysis(data_ft_rank, valid_eras, model_name, pic_fp)

        # Neutralized
        neutr_col_name = 'neutralized'
        data_ft_rank['neutralized'] = neutralize(valid_data,
                                                 data_ft_rank[model_name],
                                                 proportion=1.0)

        data_ft_rank['neutralized'] = rank_pred(data_ft_rank['neutralized'])

        print("data_ft_rank['neutralized']: ", data_ft_rank['neutralized'])
        pic_fp = strat_dir + '/' + cluster + '/' + model_name + '_ft_n_exp.png'
        ft_exp_analysis(data_ft_rank, valid_eras, neutr_col_name, pic_fp)

    return


def pred_diagnostics(strat_dir):

    strat_c_fp = strat_dir + '/' + STRAT_CONSTITUTION_FILENAME
    strat_c = StratConstitution(strat_c_fp)
    strat_c.load()

    valid_eras = get_eras(VALID_TYPE)

    valid_data = load_valid_cl_ft_target(valid_eras)

    pred_name = AGGR_PREFIX + '13'

    pred_fp = strat_dir + '/' + PREDICTIONS_FILENAME + VALID_TYPE + '_fst.csv'

    cl_m_rank = load_data(pred_fp, cols=['id', pred_name])

    data_ft_rank = pd.concat([valid_data, cl_m_rank], axis=1)

    pic_fp = strat_dir + '/' + pred_name
    pic_fp = strat_dir + '/' + pred_name + '_ft_exp.png'
    ft_exp_analysis(data_ft_rank, valid_eras, pred_name, pic_fp)

    # Neutralized
    print("\n********************")
    print("   Neutralization")
    neutr_col_name = 'neutralized'
    data_ft_rank['neutralized'] = neutralize(data_ft_rank, pred_name)

    pic_fp = strat_dir + '/' + pred_name + '_ft_n_exp.png'
    ft_exp_analysis(data_ft_rank, valid_eras, neutr_col_name, pic_fp)

    # Ranking does not change much
    print("\n********************")
    print("   Rank")
    data_ft_rank['neutralized'] = rank_pred(data_ft_rank['neutralized'])

    pic_fp = strat_dir + '/' + pred_name + '_ft_n_r_exp.png'
    ft_exp_analysis(data_ft_rank, valid_eras, neutr_col_name, pic_fp)

    # Neutralize for each era
    print("\n********************")
    print("   Era ")
    era_neutr_col_name = 'era_neutralized'
    data_ft_rank[era_neutr_col_name] = data_ft_rank.groupby('era').apply(
        lambda x: neutralize(x, pred_name)).values

    data_ft_rank[era_neutr_col_name] = rank_pred(
        data_ft_rank[era_neutr_col_name])
    pic_fp = strat_dir + '/' + pred_name + '_ft_e_n_r_exp.png'
    ft_exp_analysis(data_ft_rank, valid_eras, era_neutr_col_name, pic_fp)

    return
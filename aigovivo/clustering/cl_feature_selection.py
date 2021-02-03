import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..data_analysis import ft_target_corr
from ..reader import load_h5_eras

# CHOICE
# start : cluster
# CL_THRESHOLD_FT_T_CORR = 0.10
# CL_THRESHOLD_FT_SCORE = 0.30

# # start : cluster_2
CL_THRESHOLD_FT_SCORE = 0.50


def plot_fts_score_t_corr(fts_scores_df, ft_t_corr):
    full_df = pd.concat([fts_scores_df, ft_t_corr], axis=1)

    full_df.loc[:, 0].plot.bar(x='ft', y='ft_score', rot=0, color='r')
    full_df.loc[:, 1].plot.bar(x='ft',
                               y='ft_t_corr',
                               rot=0,
                               color='b',
                               secondary_y=True)
    print("num_ft: ", len(fts_scores_df.index))
    plt.show()


def cl_feature_selection(era_l, cl_dict, data_fp):

    for _, cl_caract in cl_dict.items():
        cl_fts_scores = [
            era_l.era_i_j_ft_score[era_i][era_j]
            for era_i in cl_caract['eras_idx']
            for era_j in cl_caract['eras_idx']
        ]
        cl_fts_scores_df = pd.concat(cl_fts_scores, axis=1)

        # CHOICE
        # aggregate score fts as sum overall cross era_i/era_j score
        # why not use ft corr along all eras in cluster as criterai of selection ?
        cl_fts_full_scores_df = cl_fts_scores_df.sum(axis=1)

        # /!\ cl['full_score'] / cl_num_eras² and cl_caract['score']
        # should be equal
        # float() instead of float32 here for json.dump
        cl_caract['full_score'] = float(cl_fts_full_scores_df.sum())
        cl_fts_full_scores_df = cl_fts_full_scores_df / cl_caract['full_score']

        cl_fts_full_scores_df = cl_fts_full_scores_df.sort_values(
            ascending=False)

        # need to calculate corr for span of eras in cluster
        data_cl_eras_df = load_h5_eras(data_fp, cl_caract['eras_name'])
        # data_cl_eras_df = data_df.loc[data_df['era'].isin(
        #    cl_caract['eras_name'])]
        era_ft_t_corr = ft_target_corr(
            data_cl_eras_df, cl_fts_full_scores_df.index.tolist()).abs()
        # plot_fts_score_t_corr(cl_fts_full_scores_df, era_ft_t_corr)

        # First, select n best ft by target corr
        era_ft_t_corr = era_ft_t_corr.sort_values(ascending=False)

        # Then, fill up according to ft score
        cumul_ft_score = 0
        cl_cumul_ft_score = pd.Series(index=era_ft_t_corr.index,
                                      dtype=np.float32)
        for ft in era_ft_t_corr.index:
            cumul_ft_score += cl_fts_full_scores_df.loc[ft]
            cl_cumul_ft_score[ft] = cumul_ft_score

        cl_sel_ft = cl_cumul_ft_score.loc[lambda x: x < CL_THRESHOLD_FT_SCORE]
        # print("cl_sel_ft: ", cl_sel_ft.size)
        # plot_fts_score_t_corr(era_ft_t_corr, cl_cumul_ft_score)

        cl_caract['mean_t_corr'] = era_ft_t_corr[
            cl_sel_ft.index.tolist()].mean()

        cl_caract['selected_features'] = cl_sel_ft.index.tolist()

    return cl_dict

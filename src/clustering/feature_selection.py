
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from corr_analysis import ft_target_corr

# CHOICE
CL_THRESHOLD_FT_T_CORR = 0.33
CL_THRESHOLD_FT_SCORE = 0.66


def plot_fts_score_t_corr(fts_scores_df, ft_t_corr):
    full_df = pd.concat([fts_scores_df, ft_t_corr], axis=1)

    full_df.loc[:, 0].plot.bar(x='ft', y='ft_score', rot=0, color='r')
    full_df.loc[:, 1].plot.bar(
        x='ft', y='ft_t_corr', rot=0, color='b', secondary_y=True)
    print("num_ft: ", len(fts_scores_df.index))
    plt.show()


def feature_selection(era_l, cl_dict, data_df):

    for _, cl_caract in cl_dict.items():
        cl_fts_scores = [era_l.era_i_j_ft_score[era_i][era_j]
                         for era_i in cl_caract['eras_idx']
                         for era_j in cl_caract['eras_idx']]
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
        data_cl_eras_df = data_df.loc[data_df['era'].isin(
            cl_caract['eras_name'])]
        era_ft_t_corr = ft_target_corr(
            data_cl_eras_df, cl_fts_full_scores_df.index.tolist()).abs()
        #plot_fts_score_t_corr(cl_fts_full_scores_df, era_ft_t_corr)

        # First, select n first by target corr
        era_ft_t_corr = era_ft_t_corr.sort_values(
            ascending=False)
        th_t_corr_idx = int(era_ft_t_corr.size * CL_THRESHOLD_FT_T_CORR)

        # Then, fill up according to ft score
        idx_list = era_ft_t_corr.index.tolist(
        )[:th_t_corr_idx] + cl_fts_full_scores_df.index.tolist()
        new_order_idx = []
        _ = [new_order_idx.append(
            idx) for idx in idx_list if idx not in new_order_idx]
        cl_fts_full_scores_df_new = cl_fts_full_scores_df.reindex(
            new_order_idx)
        era_ft_t_corr_new = era_ft_t_corr.reindex(new_order_idx)

        # here, reverse order of arguments
        # plot_fts_score_t_corr(era_ft_t_corr_new, cl_fts_full_scores_df_new)

        cumul_ft_score = 0
        cl_cumul_ft_score = pd.Series(
            index=cl_fts_full_scores_df_new.index, dtype=np.float32)
        for ft in cl_fts_full_scores_df_new.index:
            cumul_ft_score += cl_fts_full_scores_df_new.loc[ft]
            cl_cumul_ft_score[ft] = cumul_ft_score

        #plot_fts_score_t_corr(cl_cumul_ft_score, era_ft_t_corr_new)

        cl_selected_ft = cl_cumul_ft_score.loc[lambda x: x <
                                               CL_THRESHOLD_FT_SCORE]

        cl_caract['mean_t_corr'] = era_ft_t_corr_new[
            cl_selected_ft.index.tolist()].mean()

        cl_caract['selected_features'] = cl_selected_ft.index.tolist()

    return cl_dict

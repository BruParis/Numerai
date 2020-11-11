from reader_csv import ReaderCSV
from era_comparator import EraComparator
import json
import pylab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import scipy
import os
import errno
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
sns.set_theme(color_codes=True)

CORR_THRESHOLD = 0.036
MIN_NUM_ERAS = 6
# MIN_NUM_ERAS = 3
CL_THRESHOLD_FT_SCORE = 0.80
ERA_CL_DIRNAME = 'data_clusters'


def plot_mat(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    max_val = np.amax(data)
    min_val = np.amin(data)
    cax = ax.matshow(data, vmax=max_val, vmin=min_val)
    fig.colorbar(cax)
    plt.show()


def plot_matrix_clustering(era_comp, era_cl_dict, save=False):

    num_eras, _ = era_comp.era_score_mat.shape

    # plot dendrogram.
    fig = pylab.figure()
    ax1 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
    dend = sch.dendrogram(era_comp.linkage)  # needed here to plot dendogram
    ax1.set_xticks([])
    ax1.set_yticks([])

    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])

    im = axmatrix.matshow(era_comp.era_score_reordered_mat,
                          aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    im_shape = im.get_extent()
    era_length = (im_shape[1] - im_shape[0]) / num_eras
    for _, cl in era_cl_dict.items():
        eras_idxs = cl['eras']
        eras_dend_idx = [era_comp.era_to_dend_idx[e] for e in eras_idxs]
        square_min = era_length * min(eras_dend_idx)
        square_max = era_length * max(eras_dend_idx)
        square_dim = square_max - square_min + 1

        edge_color = 'g' if cl['small'] else 'b'
        rect = patches.Rectangle(
            (square_min + im_shape[0], square_min + im_shape[0]),
            square_dim, square_dim, linewidth=2, edgecolor=edge_color, facecolor='none')
        axmatrix.add_patch(rect)

    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
    pylab.colorbar(im, cax=axcolor)
    plt.show()

    if save:
        fig.savefig(ERA_CL_DIRNAME + '/dendrogram.png')

    return


def measure_score_prox(era_i_t_corr, era_j_t_corr):
    # CHOICE
    # Use product as measure of score & proximity between eras
    # strong corr. in era i will be nullified if weak in j and
    # hence cluster strongest together, instead of using just dist.
    # which will only expose similarity

    res = era_i_t_corr * era_j_t_corr
    # res = np.linalg.norm(era_i_t_corr - era_j_t_corr)

    # yield res.sum()
    return res.sum()
    # return res


def find_value_indexes(data_l, value):
    res = [i for i in range(len(data_l)) if data_l[i] == value]
    return res


def find_fst_idx_value(data_l, value):
    for i in range(len(data_l)):
        if data_l[i] == value:
            return i
    return -1


def caracterize_cl(era_comp, cl):
    cl['eras_dend_idx'] = [era_comp.era_to_dend_idx[era] for era in cl['eras']]
    cl['score'] = era_comp.cluster_ponderated_score(cl['eras_dend_idx'])
    cl['small'] = len(cl['eras']) < MIN_NUM_ERAS
    cl['full_score'] = None
    cl['selected_features'] = None
    return cl


def set_cl_dict(era_comp, era_cluster_idx):
    clusters = set(era_cluster_idx)
    cl_dict = {str(cl): {'eras': find_value_indexes(
        era_cluster_idx, cl)} for cl in clusters}
    cl_dict = {cl: caracterize_cl(era_comp, v) for cl, v in cl_dict.items()}

    return cl_dict


def has_small_cl(cl_dict):
    for _, v in cl_dict.items():
        if v['small']:
            return True
    return False


def aggregate_small_clusters(era_comp, cl_dict):

    while has_small_cl(cl_dict):
        del_s_cl = []
        for s_cl, s_cl_caract in cl_dict.items():
            if not s_cl_caract['small']:
                continue

            min_s_cl_era = min(s_cl_caract['eras_dend_idx'])
            max_s_cl_era = max(s_cl_caract['eras_dend_idx'])

            left_cl = None
            right_cl = None
            for cl_aux, cl_aux_caract in cl_dict.items():
                if s_cl == cl_aux:
                    continue

                if cl_aux in del_s_cl:
                    continue

                min_cl_aux_era = min(cl_aux_caract['eras_dend_idx'])
                max_cl_aux_era = max(cl_aux_caract['eras_dend_idx'])

                if left_cl is None and min_s_cl_era == max_cl_aux_era + 1:
                    left_cl = cl_aux
                elif right_cl is None and max_s_cl_era == min_cl_aux_era - 1:
                    right_cl = cl_aux

                if left_cl is not None and right_cl is not None:
                    break

            if left_cl is None and right_cl is None:
                continue

            b_go_left = left_cl is not None
            if b_go_left and right_cl is not None:
                l_dist_score = abs(cl_dict.get(left_cl)[
                                   'score'] - s_cl_caract['score'])
                r_dist_score = abs(cl_dict.get(right_cl)[
                                   'score'] - s_cl_caract['score'])
                b_go_left = l_dist_score < r_dist_score

            agg_cl = cl_dict[left_cl] if b_go_left else cl_dict[right_cl]
            agg_cl['eras'] = agg_cl['eras'] + s_cl_caract['eras']
            caracterize_cl(era_comp, agg_cl)

            del_s_cl.append(s_cl)

        for cl in del_s_cl:
            del cl_dict[cl]

    plot_matrix_clustering(era_comp, cl_dict, save=True)
    return cl_dict


def eras_clustering(era_comp):

    era_comp.compute_eras_linkage()

    # seaborn to compare methods
    era_corr_t_corr_clusters = sns.clustermap(era_comp.era_score_mat)
    era_corr_t_corr_clusters.savefig(ERA_CL_DIRNAME + '/matrix_clustering.png')

    # t_l = [0.07, 0.5, 0.94071673, 0.95, 1, 1.01, 1.02, 1.023]
    # t_l = [0.01, 0.06, 0.07, 0.08, 0.09, 1, 1.2, 1.3, 1.4]
    # t_l = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # for crit in t_l:
    #     era_cluster_idx = fcluster(
    #         era_comp.linkage, criterion='inconsistent', t=crit)
    #     clusters = set(era_cluster_idx)
    #     era_cl_dict = {cl: find_value_indexes(
    #         era_cluster_idx, cl) for cl in clusters}
    #     plot_matrix_clustering(era_comp, era_cl_dict)

    criteria = 0.95
    #criteria = 0.45
    era_cluster_idx = fcluster(
        era_comp.linkage, criterion='inconsistent', t=criteria)
    era_cl_dict = set_cl_dict(era_comp, era_cluster_idx)

    plot_matrix_clustering(era_comp, era_cl_dict, save=True)

    full_era_cl_dict = aggregate_small_clusters(era_comp, era_cl_dict)

    return full_era_cl_dict


def plot_fts_score(fts_scores_df):
    # plt.hist(cl_fts_scores_df.values)
    fts_scores_df.plot.bar(x='ft', y='ft_score', rot=0, color='r')
    print("num_ft: ", len(fts_scores_df.index))
    plt.show()


def cluster_ft_selection(era_comp, cl_dict):

    for _, cl_caract in cl_dict.items():
        cl_fts_scores = [era_comp.era_i_j_ft_score[era_i][era_j]
                         for era_i in cl_caract['eras']
                         for era_j in cl_caract['eras']]
        cl_fts_scores_df = pd.concat(cl_fts_scores, axis=1)

        # CHOICE
        # aggregate score fts as sum overall cross era_i/era_j score
        cl_fts_full_scores_df = cl_fts_scores_df.sum(axis=1)

        # /!\ cl['full_score'] / cl_num_eras² and cl_caract['score']
        # should be equal
        # float() instead of float32 here for json.dump
        cl_caract['full_score'] = float(cl_fts_full_scores_df.sum())
        cl_fts_full_scores_df = cl_fts_full_scores_df / cl_caract['full_score']

        cl_fts_full_scores_df = cl_fts_full_scores_df.sort_values(
            ascending=False)

        # plot_fts_score(cl_fts_full_scores_df)

        cumul_ft_score = 0
        cl_cumul_ft_score = pd.Series(index=cl_fts_full_scores_df.index)
        for ft in cl_fts_full_scores_df.index:
            cumul_ft_score += cl_fts_full_scores_df.loc[ft]
            cl_cumul_ft_score[ft] = cumul_ft_score

        # plot_fts_score(cl_cumul_ft_score)

        cl_selected_ft = cl_cumul_ft_score.loc[lambda x: x <
                                               CL_THRESHOLD_FT_SCORE]

        cl_caract['selected_features'] = cl_selected_ft.index.tolist()

        # plot_fts_score(cl_selected_ft)

    return cl_dict


def makedir_cluster(era_comp, era_cl_dict, corr_data_idx):

    try:
        os.makedirs(ERA_CL_DIRNAME)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Error with : make dir ", ERA_CL_DIRNAME)
            exit(1)

    era_cl_filepath = ERA_CL_DIRNAME + '/era_clusters.json'
    with open(era_cl_filepath, 'w') as f:
        json.dump(era_cl_dict, f, indent=4)

    era_cross_t_corr_df = pd.DataFrame(
        era_comp.era_score_mat, columns=corr_data_idx, index=corr_data_idx)

    era_cross_t_corr_filepath = ERA_CL_DIRNAME + '/era_cross_t_corr.csv'
    era_cross_t_corr_df.to_csv(era_cross_t_corr_filepath)


def main():

    era_ft_corr_filename = "eras_ft_target_corr.csv"

    file_reader = ReaderCSV(era_ft_corr_filename)
    corr_data_df = file_reader.read_csv().set_index("era")

    #corr_data_df = corr_data_df.iloc[0: 40]

    # era_i/js must all have same features
    era_comp = EraComparator(corr_data_df)

    plot_mat(era_comp.era_score_mat)

    era_cl_dict = eras_clustering(era_comp)

    cluster_ft_selection(era_comp, era_cl_dict)

    corr_data_idx = corr_data_df.index
    makedir_cluster(era_comp, era_cl_dict, corr_data_idx)


if __name__ == '__main__':
    main()

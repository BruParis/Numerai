from .era_linkage import EraLinkage
from .cl_feature_selection import cl_feature_selection
from ..common import *
from ..reader import ReaderCSV
from ..strat import ModelConstitution

import os
import errno
import pylab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
sns.set_theme(color_codes=True)

MIN_NUM_ERAS = 3


def plot_mat(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    max_val = np.amax(data)
    min_val = np.amin(data)
    cax = ax.matshow(data, vmax=max_val, vmin=min_val)
    fig.colorbar(cax)
    plt.show()


def plot_matrix_clustering(era_l, era_cl_dict, bSave=False, bShow=True):

    num_eras, _ = era_l.era_score_mat.shape

    # plot dendrogram.
    fig = pylab.figure()
    ax1 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
    dend = sch.dendrogram(era_l.linkage)  # needed here to plot dendogram
    ax1.set_xticks([])
    ax1.set_yticks([])

    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])

    im = axmatrix.matshow(era_l.era_score_reordered_mat,
                          aspect='auto',
                          origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    im_shape = im.get_extent()
    era_length = (im_shape[1] - im_shape[0]) / num_eras
    for _, cl in era_cl_dict.items():
        eras_idxs = cl['eras_idx']
        eras_dend_idx = [era_l.era_to_dend_idx[e] for e in eras_idxs]
        square_min = era_length * min(eras_dend_idx)
        square_max = era_length * max(eras_dend_idx)
        square_dim = square_max - square_min + 1

        edge_color = 'g' if cl['small'] else 'b'
        rect = patches.Rectangle(
            (square_min + im_shape[0], square_min + im_shape[0]),
            square_dim,
            square_dim,
            linewidth=2,
            edgecolor=edge_color,
            facecolor='none')
        axmatrix.add_patch(rect)

    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
    pylab.colorbar(im, cax=axcolor)

    if bShow:
        plt.show()

    if bSave:
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


def caracterize_cl(era_l, cl):
    cl['eras_name'] = ['era' + str(era_idx + 1) for era_idx in cl['eras_idx']]
    cl['eras_dend_idx'] = [
        era_l.era_to_dend_idx[era_idx] for era_idx in cl['eras_idx']
    ]
    cl['score'] = era_l.cluster_weighted_score(cl['eras_dend_idx'])
    cl['small'] = len(cl['eras_idx']) < MIN_NUM_ERAS
    cl['full_score'] = None
    cl['selected_features'] = None
    return cl


def set_cl_dict(era_l, era_cluster_idx):
    clusters = set(era_cluster_idx)
    cl_dict = {
        'cluster_' + str(cl): {
            'eras_idx': find_value_indexes(era_cluster_idx, cl)
        }
        for cl in clusters
    }
    cl_dict = {cl: caracterize_cl(era_l, v) for cl, v in cl_dict.items()}

    return cl_dict


def has_small_cl(cl_dict):
    for _, v in cl_dict.items():
        if v['small']:
            return True
    return False


def aggregate_small_clusters(era_l, cl_dict):

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
                l_dist_score = abs(
                    cl_dict.get(left_cl)['score'] - s_cl_caract['score'])
                r_dist_score = abs(
                    cl_dict.get(right_cl)['score'] - s_cl_caract['score'])
                b_go_left = l_dist_score < r_dist_score

            agg_cl = cl_dict[left_cl] if b_go_left else cl_dict[right_cl]
            agg_cl['eras_idx'] = agg_cl['eras_idx'] + s_cl_caract['eras_idx']
            caracterize_cl(era_l, agg_cl)

            del_s_cl.append(s_cl)

        for cl in del_s_cl:
            del cl_dict[cl]

    #plot_matrix_clustering(era_l, cl_dict, bSave=True)
    return cl_dict


def eras_clustering(era_l):

    era_l.compute_eras_linkage()

    # seaborn to compare methods
    era_corr_t_corr_clusters = sns.clustermap(era_l.era_score_mat)
    era_corr_t_corr_clusters.savefig(ERA_CL_DIRNAME + '/matrix_clustering.png')

    # t_l = [0.07, 0.5, 0.94071673, 0.95, 1, 1.01, 1.02, 1.023]
    # t_l = [0.01, 0.06, 0.07, 0.08, 0.09, 1, 1.2, 1.3, 1.4]
    # t_l = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # for crit in t_l:
    #     era_cluster_idx = fcluster(
    #         era_l.linkage, criterion='inconsistent', t=crit)
    #     clusters = set(era_cluster_idx)
    #     era_cl_dict = {cl: find_value_indexes(
    #         era_cluster_idx, cl) for cl in clusters}
    #     plot_matrix_clustering(era_l, era_cl_dict)

    # criteria_l = np.linspace(start=0.05, stop=1.95, num=20)
    # for cr in criteria_l:
    #     print("cr: ", cr)
    #     era_cluster_idx = fcluster(
    #         era_l.linkage, criterion='inconsistent', t=cr)
    #     era_cl_dict = set_cl_dict(era_l, era_cluster_idx)
    #     plot_matrix_clustering(era_l, era_cl_dict)
    # exit(0)

    criteria = 0.95
    # criteria = 0.45
    era_cluster_idx = fcluster(era_l.linkage,
                               criterion='inconsistent',
                               t=criteria)
    era_cl_dict = set_cl_dict(era_l, era_cluster_idx)

    plot_matrix_clustering(era_l, era_cl_dict, bSave=True)

    full_era_cl_dict = aggregate_small_clusters(era_l, era_cl_dict)

    #plot_matrix_clustering(era_l, full_era_cl_dict, bSave=True, bShow=False)
    plot_matrix_clustering(era_l, full_era_cl_dict, bSave=False, bShow=True)

    return full_era_cl_dict


def save_clusters(model_c, era_l, era_cl_dict, corr_data_idx):

    model_c.clusters = era_cl_dict
    model_c.save()

    era_cross_score_df = pd.DataFrame(era_l.era_score_mat,
                                      columns=corr_data_idx,
                                      index=corr_data_idx)

    era_cross_score_df.to_csv(ERA_CROSS_SCORE_FP)


def make_cl_dir():
    bDirAlready = False
    try:
        os.makedirs(ERA_CL_DIRNAME)
    except OSError as e:
        bDirAlready = e.errno == errno.EEXIST
        if not bDirAlready:
            print("Error with : make dir ", ERA_CL_DIRNAME)
            exit(1)

    if not bDirAlready:
        cl_c_filename = ERA_CL_DIRNAME + '/model_constitution.json'
        cl_c = ModelConstitution(cl_c_filename)
        cl_c.eras_ft_t_corr_file = ERAS_FT_T_CORR_FP
        cl_c.save()


def clustering(dirname):
    print(" -> clustering")

    make_cl_dir()

    model_c = ModelConstitution(dirname + '/' + MODEL_CONSTITUTION_FILENAME)
    model_c.load()

    file_reader = ReaderCSV(model_c.eras_ft_t_corr_file)
    corr_data_df = file_reader.read_csv().set_index("era")
    # corr_data_df = corr_data_df.iloc[0: 40]
    # era_i/js must all have same features
    era_l = EraLinkage(corr_data_df)
    # plot_mat(era_l.era_score_mat)
    era_cl_dict = eras_clustering(era_l)

    cl_feature_selection(era_l, era_cl_dict, TRAINING_STORE_H5_FP)

    save_clusters(model_c, era_l, era_cl_dict, corr_data_df.index)

from ..common import *
from ..reader import ReaderCSV, load_h5_eras
from ..data_analysis import feature_t_corr
from ..strat import ModelConstitution

import os
import errno
import pylab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(color_codes=True)

# CHOICE
CL_MIN_RATIO_FT_T_CORR = 0.10
CL_THRESHOLD_FT_T_CORR = 0.04


def load_eras():
    file_reader = ReaderCSV(TRAINING_DATA_FP)
    eras_df = file_reader.read_csv(columns=['id', 'era']).set_index('id')

    eras = eras_df.era.unique()

    return eras


def load_features():
    file_reader = ReaderCSV(TRAINING_DATA_FP)
    data_header = file_reader.read_csv_header()

    features = [c for c in data_header if c.startswith("feature")]
    return features


def find_value_indexes(data_l, value):
    res = [i for i in range(len(data_l)) if data_l[i] == value]
    return res


def find_fst_idx_value(data_l, value):
    for i in range(len(data_l)):
        if data_l[i] == value:
            return i
    return -1


def set_cl_dict(era_l, era_cluster_idx):
    clusters = set(era_cluster_idx)
    cl_dict = {
        'cluster_' + str(cl): {
            'eras_idx': find_value_indexes(era_cluster_idx, cl)
        }
        for cl in clusters
    }
    cl_dict = {cl: {'selected_features': None} for cl, v in cl_dict.items()}

    return cl_dict


def plot_fts_t_corr(ft_t_corr):

    ft_t_corr.plot.bar(x='ft', y='ft_score', rot=0, color='r')
    plt.show()


def eras_clustering(era_l, features):
    file_reader = ReaderCSV(TRAINING_DATA_FP)
    era_dict = dict()

    for era in era_l:
        print("era: ", era)
        data_era_df = file_reader.read_csv_matching('era',
                                                    [era]).set_index('id')

        ft_t_corr = feature_t_corr(data_era_df,
                                   features).abs().sort_values(ascending=False)

        # plot_fts_t_corr(ft_t_corr)
        total_num_ft = len(ft_t_corr)
        min_len = int(total_num_ft * CL_MIN_RATIO_FT_T_CORR)

        # select ft with correlation above ratio
        sel_ft_t_corr = ft_t_corr[ft_t_corr > CL_THRESHOLD_FT_T_CORR]

        # plot_fts_t_corr(sel_ft_t_corr)

        # Complete if min feature ratio not reached
        if sel_ft_t_corr.size < min_len:
            sel_ft_t_corr = ft_t_corr.head(min_len)
        # plot_fts_t_corr(ft_t_corr)

        th_t_corr_idx = int(ft_t_corr.size * CL_THRESHOLD_FT_T_CORR)
        selected_ft_t_corr = ft_t_corr.head(th_t_corr_idx)

        era_dict[era] = dict()
        era_dict[era]['eras_name'] = [era]
        era_dict[era]['mean_t_corr'] = selected_ft_t_corr.mean()
        era_dict[era]['selected_features'] = selected_ft_t_corr.index.tolist()

    return era_dict


def make_cl_dir(strat_dir):
    bDirAlready = False
    try:
        os.makedirs(strat_dir)
    except OSError as e:
        bDirAlready = e.errno == errno.EEXIST
        if not bDirAlready:
            print("Error with : make dir ", strat_dir)
            exit(1)

    if not bDirAlready:
        cl_c_filename = strat_dir + '/model_constitution.json'
        cl_c = ModelConstitution(cl_c_filename)
        cl_c.eras_ft_t_corr_file = ERAS_FT_T_CORR_FP
        cl_c.save()


def simple_era_clustering(strat_dir):
    print(" -> simple era clustering")

    make_cl_dir(strat_dir)

    model_c = ModelConstitution(strat_dir + '/' + MODEL_CONSTITUTION_FILENAME)
    model_c.load()

    era_l = load_eras()
    fts = load_features()

    era_cl_dict = eras_clustering(era_l, fts)

    model_c.clusters = era_cl_dict
    model_c.save()

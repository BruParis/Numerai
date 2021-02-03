from sklearn import feature_selection as f_s
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os

from ..common import *
from ..reader import ReaderCSV

MI_PLOT_THRESHOLD = 0.006
MI_THRESHOLD = 0.025


def ft_target_mut_i(mut_i_fp):

    f_r = ReaderCSV(TRAINING_DATA_FP)
    train_data = f_r.read_csv().set_index('id')

    eras_l = train_data.era.unique()
    ft_l = [col for col in train_data.columns if col.startswith('feature_')]

    res = pd.DataFrame(index=eras_l, columns=ft_l)
    res.index.name = 'era'

    for era in eras_l:
        print("era: ", era)
        era_data = train_data.loc[train_data.era == era]
        era_target = era_data.loc[:, TARGET_LABEL]
        era_ft_data = era_data.loc[:, ft_l]

        era_ft_t_mut_i = f_s.mutual_info_regression(
            era_ft_data,
            era_target,
        )

        res.loc[era] = era_ft_t_mut_i
    print("res: ", res)

    with open(mut_i_fp, 'w') as fp:
        res.to_csv(fp)

    return


def plot_matrix(matrix_mut, save_fp):
    print("matrix_mut: ", matrix_mut)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    max_m = ax.matshow(matrix_mut)
    fig.colorbar(max_m)
    # fig = pylab.figure()
    # axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    # axmatrix.matshow(matrix_mut, aspect='auto', origin='lower')

    plt.show()
    fig.savefig(save_fp)


def plot_hist(strat_dir, data_df, bSave=False):
    data_array = data_df.values.flatten()
    data_array_f = [mut_i for mut_i in data_array if mut_i > MI_PLOT_THRESHOLD]
    plt.hist(data_array_f, bins=100)
    plt.show()
    plt.savefig(strat_dir + '/selected_ft_mut_i.png')


def ft_selection(strat_dir):
    mut_i_fp = ERAS_FT_TARGET_MI_FP

    if not os.path.isfile(mut_i_fp):
        ft_target_mut_i(mut_i_fp)

    mut_r = ReaderCSV(ERAS_FT_TARGET_MI_FP)

    era_ft_t_mut_i = mut_r.read_csv().set_index('era')

    full_matrix = DATA_DIRNAME + '/eras_ft_target_mut_i.png'
    plot_matrix(era_ft_t_mut_i, full_matrix)

    plot_hist(strat_dir, era_ft_t_mut_i)

    # Filter values by minimal MI_THRESHOLD
    mut_i_filtered = era_ft_t_mut_i.loc[:, lambda x: x.max() > MI_THRESHOLD]
    print("mut_i_filtered: ", mut_i_filtered)
    plot_hist(strat_dir, mut_i_filtered, bSave=True)

    sel_ft_matrix_fp = strat_dir + '/selected_ft_mut_i.png'
    plot_matrix(mut_i_filtered, sel_ft_matrix_fp)

    mut_i_fp = strat_dir + '/' + SELECTED_FT_MUT_I
    with open(mut_i_fp, 'w') as fp:
        mut_i_filtered.to_csv(fp)

    ft_sel_dict = dict()
    ft_sel_dict['mut_i_threshold'] = MI_THRESHOLD
    ft_sel_dict['selected_features'] = mut_i_filtered.columns.tolist()

    with open(CL_SELECTED_FT_JSON, 'w') as fp:
        json.dump(ft_sel_dict, fp, indent=4)

    return

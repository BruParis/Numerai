from sklearn import feature_selection as f_s
from sklearn import metrics as ms
import pandas as pd
import numpy as np
import itertools
import json
import time

from ..common import *
from ..reader import ReaderCSV
from ..threadpool import pool_map

from multiprocessing import Pool


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def era_cross_mut_i(train_data, target_data, eras_xy, sel_ft):
    era_x = eras_xy[0]
    era_y = eras_xy[1]

    era_xy_target_data = target_data.loc[target_data['era'].isin(eras_xy)]
    era_xy_sel_data = train_data.loc[train_data['era'].isin(eras_xy)]

    # era_xy_mut_i = f_s.mutual_info_regression(
    #     era_xy_sel_data[sel_ft],
    #     era_xy_target_data[TARGET_LABEL],
    # )

    ft_t_corr = [
        np.corrcoef(era_xy_target_data[TARGET_LABEL],
                    era_xy_sel_data[ft])[0, 1] for ft in sel_ft
    ]

    #new_row = [era_x, era_y] + ['{:.8f}'.format(x) for x in era_xy_mut_i]
    new_row = [era_x, era_y] + ['{:.8f}'.format(x) for x in ft_t_corr]
    return new_row


def generate_cross_corr():

    sel_ft_dict = load_json(CL_SELECTED_FT_JSON)
    sel_ft = sel_ft_dict['selected_features']

    #DEV
    sel_ft = sel_ft[:4]

    f_r = ReaderCSV(TRAINING_DATA_FP)
    cols = ['id', 'era', TARGET_LABEL] + sel_ft
    train_data = f_r.read_csv(columns=cols).set_index('id')
    target_data = train_data.loc[:, ['era', TARGET_LABEL]]

    eras_l = train_data.era.unique()
    #DEV
    eras_l = eras_l[:4]
    eras_cross = itertools.product(eras_l, eras_l)

    pd_col = ['era_x', 'era_y'] + sel_ft
    mut_i_matrix = pd.DataFrame(columns=pd_col)
    # print("mut_i_matrix: ", mut_i_matrix)

    mut_i_arg = list(
        zip(itertools.repeat(train_data), itertools.repeat(target_data),
            eras_cross, itertools.repeat(sel_ft)))

    mut_i_matrix_data = pool_map(era_cross_mut_i,
                                 mut_i_arg,
                                 collect=True,
                                 arg_tuple=True)

    #print("mut_i_matrix_data: ", mut_i_matrix_data)
    mut_i_matrix = pd.DataFrame(mut_i_matrix_data, columns=pd_col)

    # start_time = time.time()
    # eras_cross = itertools.product(eras_l, eras_l)
    # for era_x, era_y in eras_cross:
    #     print("era_x: ", era_x, ' - era_y: ', era_y)
    #     new_row = era_cross_mut_i(train_data, target_data, (era_x, era_y),
    #                               sel_ft)
    #     # mut_i_matrix = mut_i_matrix.append(pd.DataFrame(new_row,
    #     #                                                 columns=pd_col),
    #     #                                    ignore_index=True)
    # print("--- %s seconds ---" % (time.time() - start_time))

    with open(MI_MAT_FP, 'w') as fp:
        mut_i_matrix.to_csv(fp)

    return

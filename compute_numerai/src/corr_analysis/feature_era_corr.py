import numpy as np
import pandas as pd
import json
import os
import errno
import matplotlib.pyplot as plt

from reader import ReaderCSV, load_h5_eras
from models import ModelConstitution
from common import *


def plot_corr(data):
    corr = data.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(data.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    plt.show()


def ft_target_corr(data_df, features):
    ft_t_corr = pd.Series({
        ft: np.corrcoef(data_df[TARGET_LABEL], data_df[ft])[0, 1]
        for ft in features})

    return ft_t_corr


def get_eras_corr(data_fp, eras, features):

    data_l = []
    column_l = ['era'] + features
    for era in eras:
        data_df = load_h5_eras(data_fp, [era])

        era_ft_corr = [np.corrcoef(data_df[TARGET_LABEL], data_df[ft])[0, 1]
                       for ft in features]

        row_era_corr = [era] + era_ft_corr
        data_l.append(row_era_corr)

    res = pd.DataFrame(data_l, columns=column_l).set_index('era')

    return res


def ft_simi(fts1, fts2):
    return len(set(fts1) & set(fts2))


def export_eras_ft_filter(dict_data, filepath):
    with open(filepath, 'w') as fp:
        json.dump(dict_data, fp)


def feature_era_corr(data_csv, data_h5):

    # remove useless read for features, once h5 contains 'features' table
    # just like eras
    file_reader = ReaderCSV(data_csv)
    #data_df = file_reader.read_csv().set_index("id")

    data_df = file_reader.read_csv_header

    # fts = pd.read_hdf(data_filename, 'features')

    features = [c for c in data_df if c.startswith("feature")]

    # data_df["erano"] = data_df.era.str.slice(3).astype(int)
    # eras = data_df.erano
    eras = pd.read_hdf(data_h5, H5_ERAS).tolist()
    print("eras: ", eras)

    eras_ft_target_corr = get_eras_corr(data_h5, eras, features)
    print("eras_ft_target_corr: ", eras_ft_target_corr)

    eras_ft_target_corr.to_csv(ERAS_FT_T_CORR_FP, index=True)


def feature_t_corr(data_df, features):
    ft_corr = pd.Series({ft: np.corrcoef(data_df[TARGET_LABEL], data_df[ft])[0, 1]
                         for ft in features})

    return ft_corr

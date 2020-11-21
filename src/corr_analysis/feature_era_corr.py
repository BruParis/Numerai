import numpy as np
import pandas as pd
import json
import os
import errno
import matplotlib.pyplot as plt

from reader import ReaderCSV
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


def feature_corr_target(data_df, eras, feature):

    by_era_correlation = pd.Series({
        era: np.corrcoef(data_df[TARGET_LABEL],
                         data_df[feature])[0, 1]
        for era, data_df in data_df.groupby(eras)
    }, name=feature)

    return by_era_correlation


def get_eras_min_corr(data_df, eras, ft):
    print("ft: ", ft)
    era_corr = feature_corr_target(data_df, eras, ft)

    return era_corr


def ft_simi(fts1, fts2):
    return len(set(fts1) & set(fts2))


def export_eras_ft_filter(dict_data, filepath):
    with open(filepath, 'w') as fp:
        json.dump(dict_data, fp)


def feature_era_corr(data_filename):

    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv().set_index("id")

    features = [c for c in data_df if c.startswith("feature")]

    data_df["erano"] = data_df.era.str.slice(3).astype(int)
    eras = data_df.erano

    eras_ft_target_corr = [get_eras_min_corr(
        data_df, eras, ft) for ft in features]

    eras_ft_target_corr_df = pd.concat(
        eras_ft_target_corr, axis=1, keys=[s.name for s in eras_ft_target_corr])
    print("eras_ft_target_corr_df: ", eras_ft_target_corr_df)
    eras_ft_target_corr_df.index.names = ['era']

    eras_ft_target_corr_df.to_csv(ERAS_FT_T_CORR_FP, index=True)

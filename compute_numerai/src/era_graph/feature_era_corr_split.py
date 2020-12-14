from reader_csv import ReaderCSV
from common import *
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
import json

TARGET_LABEL = 'target_kazutsugi'
CORR_THRESHOLD = [0.02, 0.036, 0.05]


def feature_corr_target(data_df, eras, feature):

    by_era_correlation = pd.Series({
        era: np.corrcoef(data_df[TARGET_LABEL],
                         data_df[feature])[0, 1]
        for era, data_df in data_df.groupby(eras)
    })

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


def era_ft_corr_filtering(eras_ft_target_corr, eras, features, corr_th, original_data_file):

    eras_ft_corr_filtered = [s.abs().where(lambda x: x > corr_th).dropna()
                             for s in eras_ft_target_corr]
    eras_min_corr = [e_ft.index for e_ft in eras_ft_corr_filtered]
    # eras_ft_target_corr.loc[eras_ft_target_corr.abs(
    # ) > corr_th].index.tolist()

    # ft_min_eras_dict = defaultdict(list)
    eras_min_ft_dict = defaultdict(list)
    for ft, ft_eras in zip(features, eras_min_corr):
        for e in ft_eras:
            # ft_min_eras_dict[ft].append(e)
            eras_min_ft_dict['era'+str(e)].append(ft)

    # print("ft_min_eras_dict: ", ft_min_eras_dict)
    # print("     ---> len: ", len(ft_min_eras_dict))

    corr_th_str = str(corr_th).replace('0.', '')
    eras_min_ft_dict['original_data_file'] = original_data_file
    export_eras_ft_filter(eras_min_ft_dict, "eras_ft_" + corr_th_str + ".json")


def main():

    # data_filename = "numerai_training_data_10000.csv"
    # data_filename = "numerai_training_data_30000.csv"
    data_filename = "numerai_training_data.csv"
    # data_filename = "reduced_training_data_095.csv"
    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv().set_index("id")

    features = [c for c in data_df if c.startswith("feature")]
    # features = [c for c in data_df if c.startswith("pca")]

    data_df["erano"] = data_df.era.str.slice(3).astype(int)
    eras = data_df.erano

    eras_ft_target_corr = [get_eras_min_corr(
        data_df, eras, ft) for ft in features]
    for corr_th in CORR_THRESHOLD:
        era_ft_corr_filtering(eras_ft_target_corr, eras,
                              features, corr_th, data_filename)


if __name__ == '__main__':
    main()

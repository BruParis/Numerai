from reader_csv import ReaderCSV
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
import json

TARGET_LABEL = 'target_kazutsugi'
CORR_THRESHOLD = 0.036


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


def main():
    data_filename = "numerai_training_data.csv"

    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv().set_index("id")

    features = [c for c in data_df if c.startswith("feature")]

    data_df["erano"] = data_df.era.str.slice(3).astype(int)
    print("data_df.era: ", data_df.era)
    exit(0)
    eras = data_df.erano

    eras_ft_target_corr = [get_eras_min_corr(
        data_df, eras, ft) for ft in features]

    eras_ft_target_corr_df = pd.concat(
        eras_ft_target_corr, axis=1, keys=[s.name for s in eras_ft_target_corr])
    print("eras_ft_target_corr_df: ", eras_ft_target_corr_df)
    eras_ft_target_corr_df.index.names = ['era']

    eras_ft_target_corr_df.to_csv('eras_ft_target_corr.csv', index=True)


if __name__ == '__main__':
    main()

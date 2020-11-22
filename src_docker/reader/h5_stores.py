import pandas as pd
from reader import ReaderCSV
from common import *

ERA_BATCH_SIZE = 32


def list_chunks(lst):
    for i in range(0, len(lst), ERA_BATCH_SIZE):
        yield lst[i:i + ERA_BATCH_SIZE]


def set_tournament_store(h5_f, reader):
    store = pd.HDFStore(h5_f)
    header_df = reader.read_csv_header()
    features_s = pd.Series([c for c in header_df if c.startswith("feature")])

    eras_df = reader.read_csv(columns=['id', 'era']).set_index('id')
    eras_list = eras_df.era.unique()

    eras_pd_s = pd.Series(eras_list)
    store[H5_ERAS] = eras_pd_s
    store[H5_FT] = features_s

    for eras_b in list_chunks(eras_list):
        print('era batch: ', eras_b)
        eras_b_data_df = reader.read_csv_matching(
            'era', eras_b).set_index('id')
        for era in eras_b:
            era_df = eras_b_data_df.loc[eras_b_data_df['era'] == era]
            store[era] = era_df


def set_train_store(h5_f, reader):
    store = pd.HDFStore(h5_f)
    data_df = reader.read_csv()
    eras_list = data_df.era.unique()

    eras_s = pd.Series(eras_list)
    features_s = pd.Series([c for c in data_df if c.startswith("feature")])
    print('eras: ', eras_list)

    store[H5_ERAS] = eras_s
    store[H5_FT] = features_s

    for era in eras_list:
        era_df = data_df.loc[data_df['era'] == era]
        store[era] = era_df


def set_h5_stores():
    csv_files = [TRAINING_DATA_FP, TOURNAMENT_DATA_FP]
    h5_files = [TRAINING_STORE_H5_FP, TOURNAMENT_STORE_H5_FP]
    csv_h5_f = list(zip(csv_files, h5_files))

    for csv_f, h5_f in csv_h5_f:

        reader = ReaderCSV(csv_f)
        if csv_f == TRAINING_DATA_FP:
            set_train_store(h5_f, reader)
        else:
            set_tournament_store(h5_f, reader)


def load_h5_eras(data_fp, eras):
    data_df_l = [pd.read_hdf(data_fp, e) for e in eras]
    res = pd.concat(data_df_l, axis=0)

    return res

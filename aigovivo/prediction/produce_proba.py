import os
import json
import pandas as pd
import time

from ..common import *
from ..data_analysis import rank_proba_models
from ..reader import ReaderCSV, load_h5_eras
from ..models import ModelType
from ..strat import StratConstitution

from .prediction_operator import PredictionOperator

ERA_BATCH_SIZE = 32
# ERA_BATCH_SIZE = 2


def load_eras_data_type():
    file_reader = ReaderCSV(TOURNAMENT_DATA_FP)
    eras_df = file_reader.read_csv(
        columns=['id', 'era', 'data_type']).set_index('id')

    return eras_df


def load_input_data(era_target):
    file_reader = ReaderCSV(TOURNAMENT_DATA_FP)
    input_data = file_reader.read_csv_matching('era',
                                               era_target).set_index('id')

    return input_data


def list_chunks(lst):
    for i in range(0, len(lst), ERA_BATCH_SIZE):
        yield lst[i:i + ERA_BATCH_SIZE]


def make_cluster_proba(strat_dir, strat_c, model_types, data_types,
                       eras_type_df, cl_list):

    pred_op = PredictionOperator(strat_dir,
                                 strat_c,
                                 model_types,
                                 bMultiProc=False)
    for d_t in data_types:
        print("     ==== DATA_TYPE proba: ", d_t, " ====")
        eras_df = eras_type_df.loc[eras_type_df['data_type'] == d_t]
        eras_list = eras_df['era'].drop_duplicates().values

        eras_batches = list_chunks(eras_list) if d_t is not VALID_TYPE else [
            eras_list
        ]

        file_w_h_d = {cl: True for cl in cl_list}

        for era_b in eras_batches:
            start_time = time.time()
            print("proba for era batch: ", era_b)

            if COMPUTE_BOOL:
                input_data = load_input_data(era_b)
            else:
                input_data = load_h5_eras(TOURNAMENT_STORE_H5_FP, era_b)
            print("--- input data loaded in %s seconds ---" %
                  (time.time() - start_time))

            if input_data.empty:
                continue

            for cl in cl_list:
                cl_proba = pred_op.make_cl_predict(input_data, cl)
                cl_rank = rank_proba_models(cl_proba, model_types)
                cl_pred = cl_rank if COMPUTE_BOOL else pd.concat(
                    [cl_proba, cl_rank], axis=1)

                fpath = strat_dir + '/' + cl + '/' + PROBA_FILENAME + d_t + '.csv'
                f_mode = 'w' if file_w_h_d[cl] else 'a'
                with open(fpath, f_mode) as f:
                    cl_pred.to_csv(f, header=file_w_h_d[cl], index=True)
                    file_w_h_d[cl] = False


def cluster_proba(strat_dir, model_types, cl=None):

    strat_c_fp = strat_dir + '/' + STRAT_CONSTITUTION_FILENAME
    strat_c = StratConstitution(strat_c_fp)
    strat_c.load()

    eras_type_df = load_eras_data_type()

    cl_list = strat_c.clusters.keys() if cl is None else [cl]
    make_cluster_proba(strat_dir, strat_c, model_types, PREDICTION_TYPES,
                       eras_type_df, cl_list)

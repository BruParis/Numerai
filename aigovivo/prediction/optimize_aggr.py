import json
import pandas as pd

from ..reader import ReaderCSV
from ..common import *


def load_data(data_filepath, cols=None):
    file_reader = ReaderCSV(data_filepath)
    input_data = file_reader.read_csv(columns=cols).set_index('id')

    return input_data


def load_pred_aggr_orig(strat_dir, aggr_orig, cl_fn):
    print("aggr_l: ", aggr_orig)

    rank_l = []

    for cl, model_n, _ in aggr_orig['cluster_models']:
        cl_fp = strat_dir + '/' + cl + '/' + cl_fn
        cl_rank = load_data(cl_fp, cols=['id', model_n])
        rank_dict = {'cluster': cl, 'model': model_n, 'rank': cl_rank}
        rank_l.append(rank_dict)

    return rank_l


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def aggr_rank(rank_l, w_dict):

    full_pred_df = pd.DataFrame(columns=['Full'])
    for rank in rank_l:
        cl = rank['cluster']
        m_n = rank['model']
        cl_m_rank = rank['rank'].copy()
        cl_m_rank.columns('Full')

        w = w_dict[cl][m_n]

        if full_pred_df.size == 0:
            full_pred_df = cl_m_rank * weight
        else:
            full_pred_df = full_pred_df + cl_m_rank * weight


def optimize_aggr(aggr, strat_dir):

    model_aggr_fp = strat_dir + '/' + MODEL_AGGREGATION_FILENAME

    aggr_dict = load_json(model_aggr_fp)
    aggr_orig_dict = aggr_dict[aggr]

    proba_cl_fn = PROBA_FILENAME + VALID_TYPE + '.csv'

    rank_orig_l = load_pred_aggr_orig(strat_dir, aggr_orig_dict, proba_cl_fn)

    w_dict = dict()
    for cl, model_n, w in aggr_orig_dict['cluster_models']:
        if cl not in w_dict.keys():
            w_dict[cl] = dict()
        w_dict[cl] = {model_n: w}
    print('w_dict: ', w_dict)

    step_rank = aggr_rank()

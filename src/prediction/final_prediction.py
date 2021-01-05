import os
import json
import pandas as pd
import numpy as np
import itertools as it

from common import *
from reader import ReaderCSV
from prediction import PredictionOperator
from data_analysis import rank_pred, valid_score
from models import ModelType

ARITHM_MEAN_RANK = 'a_mean_rank'
GEOM_MEAN_RANK = 'g_mean_rank'

METHODS = [ARITHM_MEAN_RANK, GEOM_MEAN_RANK]


def load_json(fp):
    with open(fp, 'r') as f:
        json_data = json.load(f)

        return json_data


def load_data(fp):
    file_reader = ReaderCSV(fp)
    input_data = file_reader.read_csv().set_index('id')

    return input_data


def arith_mean_rank(data_df):
    a_mean = data_df.mean(axis=1)
    a_mean_r = rank_pred(a_mean)

    return a_mean_r


def models_method_col(models, method):
    prefix_l = ''.join([mod + '_' for mod in models]) + method
    return prefix_l


def geo_mean_rank(data_df):
    g_mean = np.exp(np.log(data_df.prod(axis=1)) / data_df.notna().sum(1))
    g_mean_r = rank_pred(g_mean)

    return g_mean_r


def save_predict(dirname, data_t, layer, pred_df):
    pred_suffix = LAYER_PRED_SUFFIX[layer]
    predict_fp = dirname + '/' + FINAL_PREDICT_FILENAME + data_t + pred_suffix

    with open(predict_fp, 'w') as f:
        pred_df.to_csv(f, index=True)


def aggr_model_comb(data_df, model_comb):

    # TODO : use weight means with models valid corr as weights

    print("model_comb: ", model_comb)

    a_mean_r = arith_mean_rank(data_df[model_comb])
    g_mean_r = geo_mean_rank(data_df[model_comb])

    res = pd.concat([a_mean_r, g_mean_r], axis=1)

    res.columns = [models_method_col(model_comb, m) for m in METHODS]

    print("res: ", res)

    return res


def final_predict_layer(dirname, pred_dict, model_types, data_types_fp):

    max_comb = len(model_types)
    model_t_comb = list(
        it.chain.from_iterable([
            it.combinations(model_types, c_len)
            for c_len in range(2, max_comb + 1)
        ]))
    print("model_t_comb: ", model_t_comb)

    for data_t, fpath in data_types_fp:

        if not os.path.exists(fpath):
            print("fst layer file not found at:", fpath)
            continue

        data_df = load_data(fpath)

        print("data_df: ", data_df)

        aggr_l = [
            aggr_model_comb(data_df, list(comb)) for comb in model_t_comb
        ]

        result_vote = pd.concat(aggr_l, axis=1)
        print("result_vote: ", result_vote)

        if data_t is VALID_TYPE:
            for col in result_vote.columns:
                pred_dict[col] = valid_score(result_vote, col)

        save_predict(dirname, data_t, FST_LAYER, result_vote)


def compute_final_pred(dirname, data_types_fp, model_types, method):

    for data_t, fpath in data_types_fp:

        if not os.path.exists(fpath):
            print("fst layer file not found at:", fpath)
            continue
        col = models_method_col(model_types, method)

        data_df = load_data(fpath)
        data_model_t_df = data_df[model_types]

        if method == ARITHM_MEAN_RANK:
            res = pd.DataFrame(arith_mean_rank(data_model_t_df), columns=[col])
        elif method == GEOM_MEAN_RANK:
            res = pd.DataFrame(geo_mean_rank(data_model_t_df), columns=[col])

        save_predict(dirname, data_t, FST_LAYER, res)


def final_pred(strat_dir, l):

    model_dict_fp = strat_dir + '/' + MODEL_CONSTITUTION_FILENAME
    model_dict = load_json(model_dict_fp)

    # FST LAYER
    if 'final_pred' not in model_dict.keys():
        model_dict['final_pred'] = dict()

    model_final_dict = model_dict['final_pred']

    layer_key = FST_LAYER if l == 'fst' else SND_LAYER
    pred_suffix = PRED_FST_SUFFIX if l == 'fst' else PRED_SND_SUFFIX
    pred_fp = [
        strat_dir + '/predictions_tournament_' + d_t + pred_suffix
        for d_t in PREDICTION_TYPES
    ]
    data_types_fp = list(zip(PREDICTION_TYPES, pred_fp))

    if COMPUTE_BOOL:
        compute_model_types = ['XGBoost', 'RandomForest', 'NeuralNetwork']
        method = ARITHM_MEAN_RANK
        compute_final_pred(strat_dir, data_types_fp, compute_model_types,
                           method)
    else:
        model_types = ['XGBoost', 'RandomForest', 'NeuralNetwork']
        model_final_dict[layer_key] = dict()
        model_layer_dict = model_final_dict[layer_key]

        final_predict_layer(strat_dir, model_layer_dict, model_types,
                            data_types_fp)

    if not COMPUTE_BOOL:
        with open(model_dict_fp, 'w') as fp:
            json.dump(model_dict, fp, indent=4)

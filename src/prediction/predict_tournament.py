import os
import json
import pandas as pd
import time
import itertools as it

from common import *
from reader import ReaderCSV, load_h5_eras
from models import ModelType, ModelConstitution
from prediction import PredictionOperator
from data_analysis import models_valid_score, proba_to_target_label, rank_proba_models, rank_proba, rank_pred, valid_score

ERA_BATCH_SIZE = 32
# ERA_BATCH_SIZE = 2


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


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


def load_data(data_filepath, cols=None):
    file_reader = ReaderCSV(data_filepath)
    input_data = file_reader.read_csv(columns=cols).set_index('id')

    return input_data


def load_validation_target():
    file_reader = ReaderCSV(TOURNAMENT_DATA_FP)
    input_data = file_reader.read_csv_matching('data_type', [VALID_TYPE],
                                               columns=['id', 'era', 'target'
                                                        ]).set_index('id')

    return input_data


def load_data_by_eras(data_filepath, eras):
    file_reader = ReaderCSV(data_filepath)
    input_data = file_reader.read_csv_matching('era', eras).set_index('id')

    return input_data


def list_chunks(lst):
    for i in range(0, len(lst), ERA_BATCH_SIZE):
        yield lst[i:i + ERA_BATCH_SIZE]


def measure_cl_pred_valid_corr(model_dict, cl, model_types, cl_pred):
    cl_models_dict = model_dict['clusters'][cl]['models']
    models_valid_score(cl_models_dict, model_types, cl_pred)


def single_cluster_pred(strat, strat_dir, model_dict, model_types,
                        data_types_fp, eras_type_df, cl):

    for data_t, fpath in data_types_fp:
        eras_df = eras_type_df.loc[eras_type_df['data_type'] == data_t]
        eras_list = eras_df['era'].drop_duplicates().values

        eras_batches = list_chunks(
            eras_list) if data_t is not VALID_TYPE else [eras_list]

        pred_op = PredictionOperator(strat,
                                     strat_dir,
                                     model_dict,
                                     model_types,
                                     data_t,
                                     bMultiProcess=False)

        cl_proba_batches = pd.DataFrame()

        for era_b in eras_batches:
            start_time = time.time()
            print("prediction for era batch: ", era_b)

            if COMPUTE_BOOL:
                input_data = load_input_data(era_b)
            else:
                input_data = load_h5_eras(TOURNAMENT_STORE_H5_FP, era_b)
            print("--- %s seconds ---" % (time.time() - start_time))

            input_data_era = input_data.loc[input_data['era'].isin(era_b)]
            input_type_data = input_data_era.loc[input_data_era['data_type'] ==
                                                 data_t]

            if input_type_data.empty:
                continue

            pred_cl = pred_op.make_cl_predict(input_type_data, cl)

            cl_proba_batches = pd.concat([cl_proba_batches, pred_cl], axis=0)

        print("cl_proba_batches: ", cl_proba_batches)
        cl_rank = rank_proba_models(cl_proba_batches, model_types)

        if data_t == VALID_TYPE:
            measure_cl_pred_valid_corr(model_dict, cl, model_types, cl_rank)

        with open(fpath, 'w') as f:
            cl_rank.to_csv(f, index=True)


def aggr_rank(rank_dict, aggr_dict, data_t):

    valid_target = load_validation_target()

    aggr_pred_dict = dict()
    for aggr_id, aggr_desc in aggr_dict.items():

        full_pred_df = pd.DataFrame(columns=COL_PROBA_NAMES)
        for cl, model, weight in aggr_desc['cluster_models']:
            cl_m_rank = rank_dict[cl][model]
            cl_m_rank.columns = ['Full']
            if full_pred_df.size == 0:
                full_pred_df = cl_m_rank * weight
            else:
                full_pred_df = full_pred_df + cl_m_rank * weight

        total_w = aggr_desc['total_w']
        full_pred_df /= total_w

        aggr_pred = rank_pred(full_pred_df)
        aggr_pred_name = 'aggr_pred_' + str(aggr_id)
        aggr_pred.columns = [aggr_pred_name]
        aggr_pred_dict[aggr_id] = aggr_pred

        if data_t == VALID_TYPE:
            aggr_desc['valid_score'] = valid_score(aggr_pred, aggr_pred_name,
                                                   valid_target)

    return aggr_pred_dict


def make_prediction_fst(strat_dir, model_dict, aggr_dict, data_types_files):

    model_types = [
        ModelType.XGBoost, ModelType.RandomForest, ModelType.NeuralNetwork
    ]
    clusters_dict = model_dict['clusters']

    if 'final_pred' not in model_dict:
        model_dict['final_pred'] = dict()

    file_w_h = {d_t: True for d_t, *_ in data_types_files}
    print("file_w_h: ", file_w_h)

    for data_t, fpath, cl_fn in data_types_files:

        rank_dict = {
            cl: {eModel.name: pd.DataFrame()
                 for eModel in model_types}
            for cl in clusters_dict.keys()
        }

        for cl, _ in clusters_dict.items():
            cl_fp = strat_dir + '/' + cl + '/' + cl_fn

            for eModel in model_types:
                cl_rank = load_data(cl_fp, cols=['id', eModel.name])
                cl_rank = cl_rank.rename(columns={eModel.name: 'rank'})

                rank_dict[cl][eModel.name] = cl_rank

        rank_aggr_pred_dict = aggr_rank(rank_dict, aggr_dict, data_t)

        all_preds = pd.DataFrame()
        for _, rank_df in rank_aggr_pred_dict.items():
            all_preds = pd.concat([all_preds, rank_df], axis=1)

        with open(fpath, 'w') as f:
            all_preds.to_csv(f, header=file_w_h[data_t], index=True)
            file_w_h[data_t] = False


def make_compute_pred(strat_dir, aggr_desc, data_types_files):

    aggr_l = {"16": aggr_desc}
    print("aggr_l: ", aggr_l)

    load_cl_m_dict = dict()
    for cl, model, w in aggr_l['16']['cluster_models']:
        if cl not in load_cl_m_dict.keys():
            load_cl_m_dict[cl] = [model]
        else:
            load_cl_m_dict[cl].append(model)

    file_w_h = {d_t: True for d_t, *_ in data_types_files}
    for data_t, fpath, cl_fn in data_types_files:

        rank_dict = {
            cl: {model: pd.DataFrame()
                 for model in cl_models}
            for cl, cl_models in load_cl_m_dict.items()
        }

        for cl, model_n, weight in aggr_desc['cluster_models']:
            cl_fp = strat_dir + '/' + cl + '/' + cl_fn
            cl_rank = load_data(cl_fp, cols=['id', model_n])
            rank_dict[cl][model_n] = cl_rank

        rank_aggr_pred_dict = aggr_rank(rank_dict, aggr_l, data_t)
        aggr_pred = rank_aggr_pred_dict["16"]

        with open(fpath, 'w') as f:
            aggr_pred.to_csv(f, header=file_w_h[data_t], index=True)
            file_w_h[data_t] = False


def make_prediction_snd(strat, strat_dir, model_dict, data_types_fp,
                        model_types):
    for data_type, fst_layer_path, snd_layer_path in data_types_fp:

        if not os.path.exists(fst_layer_path):
            print("fst layer file not found at:", fst_layer_path)
            continue

        print("fst_layer_path: ", fst_layer_path)
        fst_layer_data = load_data(fst_layer_path)

        pred_cols = model_dict[SND_LAYER]['models']['input_columns']
        print("pred_cols: ", pred_cols)
        print("fst_layer_data col: ", fst_layer_data.columns)
        print("fst_layer_data: ", fst_layer_data)
        fst_layer_cl_data = fst_layer_data.loc[:, pred_cols]

        pred_op = PredictionOperator(strat,
                                     strat_dir,
                                     model_dict,
                                     model_types,
                                     data_type,
                                     bMultiProcess=True)

        snd_layer_pred_data = pred_op.make_snd_layer_predict(fst_layer_cl_data)
        snd_layer_rank_data = rank_proba_models(snd_layer_pred_data,
                                                model_types)

        print('snd_layer_pred_data: ', snd_layer_pred_data)
        print('snd_layer_rank_data: ', snd_layer_rank_data)

        with open(snd_layer_path, 'w') as f:
            snd_layer_rank_data.to_csv(f, index=True)


def make_cluster_predict(strat_dir, strat, cl):

    model_dict_fp = strat_dir + '/' + MODEL_CONSTITUTION_FILENAME
    model_dict = load_json(model_dict_fp)
    eras_type_df = load_eras_data_type()

    model_types = [
        ModelType.XGBoost, ModelType.RandomForest, ModelType.NeuralNetwork
    ]
    # model_types = [ModelType.NeuralNetwork]

    cl_dir = strat_dir + '/' + cl
    pred_fp = [
        cl_dir + '/' + PREDICTIONS_FILENAME + d_t + '.csv'
        for d_t in PREDICTION_TYPES
    ]
    data_types_fp = list(zip(PREDICTION_TYPES, pred_fp))

    single_cluster_pred(strat, strat_dir, model_dict, model_types,
                        data_types_fp, eras_type_df, cl)

    with open(model_dict_fp, 'w') as fp:
        json.dump(model_dict, fp, indent=4)


def make_prediction(strat_dir, strat, layer):
    print('data_types: ', PREDICTION_TYPES)

    model_dict_fp = strat_dir + '/' + MODEL_CONSTITUTION_FILENAME
    model_aggr_fp = strat_dir + '/' + MODEL_AGGREGATION_FILENAME
    model_dict = load_json(model_dict_fp)
    aggr_dict = load_json(model_aggr_fp)

    start_time = time.time()

    if 'final_pred' not in model_dict.keys():
        model_dict['final_pred'] = dict()

    if layer == 'fst':
        predictions_fst_fp = [
            strat_dir + '/' + PREDICTIONS_FILENAME + d_t + PRED_FST_SUFFIX
            for d_t in PREDICTION_TYPES
        ]
        proba_cl_fn = [
            PROBA_FILENAME + d_t + '.csv' for d_t in PREDICTION_TYPES
        ]
        data_types_files = list(
            zip(PREDICTION_TYPES, predictions_fst_fp, proba_cl_fn))
        if COMPUTE_BOOL:
            compute_aggr = aggr_dict['16']
            make_compute_pred(strat_dir, compute_aggr, data_types_files)
        else:
            make_prediction_fst(strat_dir, model_dict, aggr_dict,
                                data_types_files)

    if layer == 'snd':

        filename = PREDICTIONS_FILENAME

        filename_label = filename.replace('predictions_', 'pred_label_')
        predictions_fst_fp = [
            strat_dir + '/' + filename_label + d_t + PRED_FST_SUFFIX
            for d_t in PREDICTION_TYPES
        ]
        data_types_fst_fp = list(zip(PREDICTION_TYPES, predictions_fst_fp))

        # Snd layer
        pred_snd_fp = [
            strat_dir + '/' + PREDICTIONS_FILENAME + d_t + PRED_SND_SUFFIX
            for d_t in PREDICTION_TYPES
        ]
        data_types_snd_fp = list(
            zip(PREDICTION_TYPES, predictions_fst_fp, pred_snd_fp))

        model_types_snd = [
            ModelType.XGBoost, ModelType.RandomForest, ModelType.NeuralNetwork
        ]

        make_prediction_snd(strat, strat_dir, model_dict, data_types_snd_fp,
                            model_types_snd)

    print("--- %s seconds ---" % (time.time() - start_time))

    with open(model_dict_fp, 'w') as fp:
        json.dump(model_dict, fp, indent=4)

    with open(model_aggr_fp, 'w') as fp:
        json.dump(aggr_dict, fp, indent=4)

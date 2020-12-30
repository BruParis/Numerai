import os
import json
import pandas as pd
import time

from common import *
from reader import ReaderCSV, load_h5_eras
from models import ModelType, ModelConstitution
from prediction import PredictionOperator, ranking
from corr_analysis import models_valid_score
from .ranking import rank_proba, proba_to_target_label

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
    input_data = file_reader.read_csv_matching(
        'era', era_target).set_index('id')

    return input_data


def load_data(data_filepath):
    file_reader = ReaderCSV(data_filepath)
    input_data = file_reader.read_csv().set_index('id')

    return input_data


def list_chunks(lst):
    for i in range(0, len(lst), ERA_BATCH_SIZE):
        yield lst[i:i + ERA_BATCH_SIZE]


def measure_cl_pred_valid_corr(model_dict, cl, model_types, cl_pred):
    cl_models_dict = model_dict['clusters'][cl]['models']
    models_valid_score(cl_models_dict, model_types, cl_pred)


def single_cluster_pred(strat, strat_dir, model_dict, model_types, data_types_fp,
                        eras_type_df, cl):

    for data_t, fpath in data_types_fp:
        eras_df = eras_type_df.loc[eras_type_df['data_type'] == data_t]
        eras_list = eras_df['era'].drop_duplicates().values

        eras_batches = list_chunks(
            eras_list) if data_t is not VALID_TYPE else [eras_list]

        pred_op = PredictionOperator(
            strat, strat_dir, model_dict, model_types, data_t, bMultiProcess=False)

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
            input_type_data = input_data_era.loc[input_data_era['data_type'] == data_t]

            if input_type_data.empty:
                continue

            pred_cl = pred_op.make_cl_predict(input_type_data, cl)

            cl_proba_batches = pd.concat([cl_proba_batches, pred_cl], axis=0)

        cl_rank = rank_proba(cl_proba_batches, model_types)

        if data_t == VALID_TYPE:
            measure_cl_pred_valid_corr(model_dict, cl, model_types, cl_rank)

        with open(fpath, 'w') as f:
            cl_rank.to_csv(f, index=True)


def aggregate_model_cl_preds(eModel, cl_dict, prob_cl_dict):
    model_prob_dict = {cl: proba.loc[:, proba.columns.str.startswith(
        eModel.name)] for cl, proba in prob_cl_dict.items()}
    cl_v_corr_d = {cl: cl_dict[cl]['models'][eModel.name]
                   ['valid_score']['valid_corr_mean']
                   for cl in model_prob_dict.keys()}
    cl_w_d = {cl: cl_v_corr for cl,
              cl_v_corr in cl_v_corr_d.items() if cl_v_corr > 0}
    total_w = sum([w for cl, w in cl_w_d.items()])
    print("pred_cl_dict: ", model_prob_dict)
    full_pred_model = sum(model_prob_dict[cl] * cl_w
                          for cl, cl_w in cl_w_d.items()) / total_w
    return full_pred_model


def rank_fst_proba(fpath, file_w_h, model_types, final_pred_dict, cl_dict, data_t, proba_dict):
    rank_cl_df = pd.DataFrame()
    label_cl_df = pd.DataFrame()
    for cl, cl_proba in proba_dict.items():
        cl_rank = rank_proba(cl_proba, model_types)
        cl_rank.columns = [cl + '_' + col for col in cl_rank.columns]
        rank_cl_df = pd.concat([rank_cl_df, cl_rank], axis=1)

        # classify with target label for snd layer
        cl_label = proba_to_target_label(cl_proba, model_types)
        cl_label.columns = [cl + '_' + col for col in cl_label.columns]
        label_cl_df = pd.concat([label_cl_df, cl_label], axis=1)

    fpath_label = fpath.replace('predictions_', 'pred_label_')
    with open(fpath_label, 'w') as f:
        label_cl_df.to_csv(f, header=file_w_h[data_t], index=True)

    # TODO: aggregate cluster/models with mean methods instead of final_prediction
    aggr_pred_df = pd.DataFrame()
    for eModel in model_types:
        aggr_cl_model_pred = aggregate_model_cl_preds(
            eModel, cl_dict, proba_dict)
        aggr_pred_df = pd.concat(
            [aggr_pred_df, aggr_cl_model_pred], axis=1)
    aggr_rank_df = rank_proba(aggr_pred_df, model_types)

    # TODO : Move to final_prediction
    if data_t == VALID_TYPE:
        for eModel in model_types:
            if eModel.name not in final_pred_dict.keys():
                final_pred_dict[eModel.name] = dict()

        models_valid_score(final_pred_dict, model_types, aggr_rank_df)

    full_rank_df = pd.DataFrame()
    full_rank_df = pd.concat([rank_cl_df, aggr_rank_df], axis=1)
    print("full_rank_df: ", full_rank_df)

    with open(fpath, 'w') as f:
        full_rank_df.to_csv(f, header=file_w_h[data_t], index=True)
        file_w_h[data_t] = False


def make_prediction_fst(strat, strat_dir, model_dict, data_types_fp, eras_type_df, model_types):
    # TODO:
    # Full aggregation for all clusters here
    # -> make version for specific combinaison + aggregation methods

    if 'final_pred' not in model_dict.keys():
        model_dict['final_pred'] = dict()

    final_pred_dict = model_dict['final_pred']

    file_w_h = {d_t: True for d_t, _ in data_types_fp}
    for data_t, fpath in data_types_fp:
        cl_dict = model_dict['clusters']
        eras_df = eras_type_df.loc[eras_type_df['data_type'] == data_t]
        eras_list = eras_df['era'].drop_duplicates().values

        eras_batches = list_chunks(
            eras_list) if data_t is not VALID_TYPE else [eras_list]

        pred_op = PredictionOperator(
            strat, strat_dir, model_dict, model_types, data_t, bMultiProcess=False)

        proba_dict = {cl: pd.DataFrame() for cl in cl_dict}
        for era_b in eras_batches:
            start_time = time.time()
            print("prediction for era batch: ", era_b)

            if COMPUTE_BOOL:
                input_data = load_input_data(era_b)
            else:
                input_data = load_h5_eras(TOURNAMENT_STORE_H5_FP, era_b)
            print("--- %s seconds ---" % (time.time() - start_time))

            input_data_era = input_data.loc[input_data['era'].isin(era_b)]
            input_type_data = input_data_era.loc[input_data_era['data_type'] == data_t]

            if input_type_data.empty:
                continue

            for cl in cl_dict:
                cl_proba = pred_op.make_cl_predict(input_type_data, cl)

                if not COMPUTE_BOOL:
                    if data_t == VALID_TYPE:
                        cl_rank = rank_proba(cl_proba, model_types)
                        measure_cl_pred_valid_corr(
                            model_dict, cl, model_types, cl_rank)

                    pred_fp = strat_dir + '/' + cl + '/' + PREDICTIONS_FILENAME + data_t + '.csv'
                    with open(pred_fp, 'w') as f:
                        cl_rank.to_csv(f, index=True)

                proba_dict[cl] = pd.concat([proba_dict[cl], cl_proba], axis=0)

            rank_fst_proba(fpath, file_w_h, model_types,
                           final_pred_dict, cl_dict, data_t, proba_dict)


def make_prediction_snd(strat, strat_dir, model_dict, data_types_fp, model_types):
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

        pred_op = PredictionOperator(
            strat, strat_dir, model_dict, model_types, data_type, bMultiProcess=True)

        snd_layer_pred_data = pred_op.make_snd_layer_predict(fst_layer_cl_data)
        snd_layer_rank_data = rank_proba(snd_layer_pred_data, model_types)

        print('snd_layer_pred_data: ', snd_layer_pred_data)
        print('snd_layer_rank_data: ', snd_layer_rank_data)

        with open(snd_layer_path, 'w') as f:
            snd_layer_rank_data.to_csv(f, index=True)


def make_cluster_predict(strat_dir, strat, cl):

    model_dict_fp = strat_dir + '/' + MODEL_CONSTITUTION_FILENAME
    model_dict = load_json(model_dict_fp)
    eras_type_df = load_eras_data_type()

    model_types = [ModelType.XGBoost, ModelType.RandomForest,
                   ModelType.NeuralNetwork]

    cl_dir = strat_dir + '/' + cl
    pred_fp = [cl_dir + '/' + PREDICTIONS_FILENAME + d_t +
               '.csv' for d_t in PREDICTION_TYPES]
    data_types_fp = list(zip(PREDICTION_TYPES, pred_fp))

    single_cluster_pred(strat, strat_dir, model_dict, model_types, data_types_fp,
                        eras_type_df, cl)

    with open(model_dict_fp, 'w') as fp:
        json.dump(model_dict, fp, indent=4)


def make_prediction(strat_dir, strat, layer):
    print('data_types: ', PREDICTION_TYPES)

    model_dict_fp = strat_dir + '/' + MODEL_CONSTITUTION_FILENAME
    model_dict = load_json(model_dict_fp)

    # Fst layer
    predictions_fst_fp = [strat_dir + '/' + PREDICTIONS_FILENAME +
                          d_t + PRED_FST_SUFFIX for d_t in PREDICTION_TYPES]
    data_types_fst_fp = list(zip(PREDICTION_TYPES, predictions_fst_fp))

    start_time = time.time()
    eras_type_df = load_eras_data_type()
    print("--- %s seconds ---" % (time.time() - start_time))

    if layer == 'fst':
        # model_types_fst = [ModelType.NeuralNetwork]
        model_types_fst = [ModelType.XGBoost, ModelType.RandomForest,
                           ModelType.NeuralNetwork]
        filename = PREDICTIONS_FILENAME
        predictions_fst_fp = [strat_dir + '/' + filename +
                              d_t + PRED_FST_SUFFIX for d_t in PREDICTION_TYPES]
        data_types_fst_fp = list(zip(PREDICTION_TYPES, predictions_fst_fp))
        make_prediction_fst(strat, strat_dir, model_dict,
                            data_types_fst_fp, eras_type_df, model_types_fst)

    if layer == 'snd':

        filename = PREDICTIONS_FILENAME

        filename_label = filename.replace('predictions_', 'pred_label_')
        predictions_fst_fp = [strat_dir + '/' + filename_label +
                              d_t + PRED_FST_SUFFIX for d_t in PREDICTION_TYPES]
        data_types_fst_fp = list(zip(PREDICTION_TYPES, predictions_fst_fp))

        # Snd layer
        pred_snd_fp = [strat_dir + '/' + PREDICTIONS_FILENAME +
                       d_t + PRED_SND_SUFFIX for d_t in PREDICTION_TYPES]
        data_types_snd_fp = list(
            zip(PREDICTION_TYPES, predictions_fst_fp, pred_snd_fp))

        model_types_snd = [ModelType.XGBoost, ModelType.RandomForest,
                           ModelType.NeuralNetwork]

        make_prediction_snd(
            strat, strat_dir, model_dict, data_types_snd_fp, model_types_snd)

    print("--- %s seconds ---" % (time.time() - start_time))

    with open(model_dict_fp, 'w') as fp:
        json.dump(model_dict, fp, indent=4)

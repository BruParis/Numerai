import json
import time
import pandas as pd

from ..common import *
from ..reader import ReaderCSV
from ..strat import StratConstitution, Aggregations
from ..utils import get_eras
from ..data_analysis import rank_proba_models, rank_pred, neutralize

from .pred_diagnostics import ft_exp_analysis
from .prediction_operator import PredictionOperator

ERA_BATCH_SIZE = 32


def load_input_data(eras_l):
    file_reader = ReaderCSV(TOURNAMENT_DATA_FP)
    input_data = file_reader.read_csv_matching('era', eras_l).set_index('id')

    return input_data


def load_valid_data():

    file_reader = ReaderCSV(TOURNAMENT_DATA_FP)
    valid_data = file_reader.read_csv_matching('data_type',
                                               [VALID_TYPE]).set_index('id')

    return valid_data


def load_eras_data_type(data_type=None):
    file_reader = ReaderCSV(TOURNAMENT_DATA_FP)

    if data_type is None:
        eras_df = file_reader.read_csv(
            columns=['id', 'era', 'data_type']).set_index('id')
    else:
        eras_df = file_reader.read_csv_matching(
            'data_type', [data_type], columns=['id', 'era',
                                               'data_type']).set_index('id')

    return eras_df


def list_chunks(lst):
    for i in range(0, len(lst), ERA_BATCH_SIZE):
        yield lst[i:i + ERA_BATCH_SIZE]


def predict_era_b(folder, strat_c, compute_aggr_id, aggr_dict, input_data,
                  cl_models):

    era_ranks_d = dict()
    # make rank from proba
    for cl, model_n, weight in cl_models:

        if cl not in era_ranks_d.keys():
            era_ranks_d[cl] = dict()

        if model_n not in era_ranks_d[cl].keys():
            era_ranks_d[cl][model_n] = dict()

        eModel = ModelType[model_n]
        pred_unit_op = PredictionOperator(folder,
                                          strat_c, [eModel],
                                          bMultiProc=False)

        unit_proba = pred_unit_op.make_cl_predict(input_data, cl)
        unit_rank = rank_proba_models(unit_proba, [eModel])
        era_ranks_d[cl][model_n]['rank'] = unit_rank

    full_pred_df = pd.DataFrame(columns=['Full'])
    for cl, model_n, weight in cl_models:
        cl_m_rank = era_ranks_d[cl][model_n]['rank']
        if full_pred_df.size == 0:
            full_pred_df['Full'] = cl_m_rank[model_n] * weight
        else:
            full_pred_df['Full'] += cl_m_rank[model_n] * weight

    total_w = aggr_dict['total_w']
    full_pred_df /= total_w

    full_pred_df = rank_pred(full_pred_df)
    full_pred_df.columns = [compute_aggr_id]

    full_pred_df = pd.concat([input_data[['era', TARGET_LABEL]], full_pred_df],
                             axis=1)

    return full_pred_df


def neutr_pred(aggr_dict, era_b, compute_aggr_id, pred_df, neutr_label,
               input_data):
    neutr_valid_col = [
        ft for ft in input_data.columns if ft.startswith('feature')
    ]
    if 'sel_ft' in aggr_dict['optim_neutr'].keys():
        neutr_valid_col = aggr_dict['optim_neutr']['sel_ft'].split('|')
    neutr_valid_col += ['era', TARGET_LABEL]

    neutr_df = pd.DataFrame()
    for era in era_b:
        era_input_data = input_data.loc[input_data.era == era]
        era_pred_df = pred_df.loc[era_input_data.index]

        era_neutr = neutralize(
            pd.concat([era_input_data[neutr_valid_col], era_pred_df], axis=1),
            compute_aggr_id)
        neutr_df = pd.concat([neutr_df, era_neutr], axis=1)
    pred_df[neutr_label] = neutr_df

    return pred_df


def compute_predict(layer, cluster, folder, model_types):

    # aggr_pred_16 supposed to be the best yet

    strat_fp = folder + '/' + STRAT_CONSTITUTION_FILENAME
    strat_c = StratConstitution(strat_fp)
    strat_c.load()

    compute_aggr_id = strat_c.compute_aggr
    neutr_label = compute_aggr_id + NEUTRALIZED_SUFFIX

    model_aggr_fp = folder + '/' + MODEL_AGGREGATION_FILENAME
    full_aggr_dict = Aggregations(model_aggr_fp)
    full_aggr_dict.load()

    m_aggr_dict = full_aggr_dict.get_models_aggr(model_types)

    m_aggr_dict = full_aggr_dict.get_models_aggr(model_types)
    if compute_aggr_id not in m_aggr_dict.keys():
        model_names = [m.name for m in model_types]
        print("aggr id {} not available for models: {}".format(
            compute_aggr_id, model_names))
        return

    aggr_dict = m_aggr_dict[compute_aggr_id]

    cl_models = aggr_dict['cluster_models']

    for data_type in PREDICTION_TYPES:
        start_time_d = time.time()

        print('data_type: ', data_type)

        eras_l = get_eras(data_type)

        eras_batches = list_chunks(
            eras_l) if data_type is not VALID_TYPE else [eras_l]

        data_full_pred = pd.DataFrame()

        start_time = time.time()
        for era_b in eras_batches:
            print("era batches: ", era_b)

            start_time_l = time.time()
            input_data = load_input_data(era_b)
            print("     -> era batch loaded in %s seconds" %
                  (time.time() - start_time_l))

            start_time_p = time.time()
            era_rank_target = predict_era_b(folder, strat_c, compute_aggr_id,
                                            aggr_dict, input_data, cl_models)
            print("     -> era batch pred in %s seconds" %
                  (time.time() - start_time_p))

            if strat_c.neutralize:
                neutr_pred(aggr_dict, era_b, compute_aggr_id, era_rank_target,
                           neutr_label, input_data)

            data_full_pred = pd.concat([data_full_pred, era_rank_target],
                                       axis=0)

        print("data_full_pred: ", data_full_pred)

        # data_full_pred[neutr_label] = data_full_pred.groupby('era').apply(
        #     lambda x: neutralize(x, compute_aggr_id)).values
        pred_label = neutr_label if strat_c.neutralize else compute_aggr_id
        data_full_pred[pred_label] = rank_pred(data_full_pred[pred_label])

        print("data_full_pred: ", data_full_pred)

        print("--- %s seconds ---" % (time.time() - start_time))

        if data_type == VALID_TYPE and not COMPUTE_BOOL:
            pic_fp = folder + '/compute_ft_n_exp.png'

            valid_data = load_valid_data()
            valid_pred_data = pd.concat(
                [valid_data, data_full_pred[[pred_label]]], axis=1)

            comp_diag_dict = dict()
            comp_diag_dict[pred_label] = ft_exp_analysis(
                valid_pred_data, eras_l, pred_label, pic_fp)

            comp_diag_fp = folder + '/compute_diag.json'
            with open(comp_diag_fp, 'w') as fp:
                json.dump(comp_diag_dict, fp, indent=4)

        data_type_pred_fp = folder + '/' + COMPUTE_PREDICT_PREFIX + data_type + ".csv"

        with open(data_type_pred_fp, 'w') as fp:
            data_full_pred.to_csv(fp)

        print(" -> data type pred in %s seconds" %
              (time.time() - start_time_d))

    return

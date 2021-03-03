import json
import time
import pandas as pd

from ..common import *
from ..reader import ReaderCSV
from ..strat import StratConstitution
from ..utils import get_eras
from ..data_analysis import rank_proba_models, rank_pred, neutralize, ft_exp_analysis
from ..models import ModelType

from .prediction_operator import PredictionOperator

ERA_BATCH_SIZE = 32


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


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


def predict_era(folder, strat_c, era, compute_aggr_id, neutr_label, aggr_dict,
                input_data, cl_models):

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

    era_full_pred_df = pd.DataFrame(columns=['Full'])
    for cl, model_n, weight in cl_models:
        cl_m_rank = era_ranks_d[cl][model_n]['rank']
        if era_full_pred_df.size == 0:
            era_full_pred_df['Full'] = cl_m_rank[model_n] * weight
        else:
            era_full_pred_df['Full'] += cl_m_rank[model_n] * weight

    total_w = aggr_dict['total_w']
    era_full_pred_df /= total_w

    era_full_pred_df = rank_pred(era_full_pred_df)
    era_full_pred_df.columns = [compute_aggr_id]

    era_full_pred_df[neutr_label] = neutralize(
        pd.concat([input_data, era_full_pred_df], axis=1), compute_aggr_id)

    era_full_pred_df = pd.concat(
        [input_data[['era', TARGET_LABEL]], era_full_pred_df], axis=1)

    return era_full_pred_df


def compute_predict(layer, cluster, folder):

    # aggr_pred_16 supposed to be the best yet

    strat_fp = folder + '/' + STRAT_CONSTITUTION_FILENAME
    strat_c = StratConstitution(strat_fp)
    strat_c.load()

    compute_aggr_id = strat_c.compute_aggr
    neutr_label = compute_aggr_id + NEUTRALIZED_SUFFIX

    model_aggr_fp = folder + '/' + MODEL_AGGREGATION_FILENAME
    aggr_dict = load_json(model_aggr_fp)
    aggr_dict = aggr_dict[compute_aggr_id]

    cl_models = aggr_dict['cluster_models']

    for data_type in PREDICTION_TYPES:

        print('data_type: ', data_type)

        eras_l = get_eras(data_type)
        print('eras_l: ', eras_l)

        eras_batches = list_chunks(
            eras_l) if data_type is not VALID_TYPE else [eras_l]

        data_full_pred = pd.DataFrame()

        start_time = time.time()
        for era_b in eras_batches:

            input_data = load_input_data(era_b)

            for era in era_b:
                print('era: ', era)
                era_input_data = input_data.loc[input_data.era == era]

                era_rank_target = predict_era(folder, strat_c, era,
                                              compute_aggr_id, neutr_label,
                                              aggr_dict, era_input_data,
                                              cl_models)

                data_full_pred = pd.concat([data_full_pred, era_rank_target],
                                           axis=0)

        # data_full_pred[neutr_label] = data_full_pred.groupby('era').apply(
        #     lambda x: neutralize(x, compute_aggr_id)).values
        data_full_pred[neutr_label] = rank_pred(data_full_pred[neutr_label])

        print("data_full_pred: ", data_full_pred)

        print("--- %s seconds ---" % (time.time() - start_time))

        if data_type == VALID_TYPE and not COMPUTE_BOOL:
            pic_fp = folder + '/compute_ft_n_exp.png'

            valid_data = load_valid_data()
            valid_pred_data = pd.concat(
                [valid_data, data_full_pred[[compute_aggr_id, neutr_label]]],
                axis=1)

            comp_diag_dict = dict()
            comp_diag_dict[compute_aggr_id] = ft_exp_analysis(
                valid_pred_data, eras_l, compute_aggr_id, pic_fp)
            comp_diag_dict[neutr_label] = ft_exp_analysis(
                valid_pred_data, eras_l, neutr_label, pic_fp)

            comp_diag_fp = folder + '/compute_diag.json'
            with open(comp_diag_fp, 'w') as fp:
                json.dump(comp_diag_dict, fp, indent=4)

        data_type_pred_fp = folder + '/' + COMPUTE_PREDCIT_PREFIX + data_type + ".csv"

        with open(data_type_pred_fp, 'w') as fp:
            data_full_pred.to_csv(fp)

    return

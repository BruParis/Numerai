import json
import time
import pandas as pd

from ..common import *
from ..reader import ReaderCSV, load_h5_eras
from ..utils import get_eras
from ..data_analysis import neutralize, rank_pred
from ..strat import Aggregations


def load_data(data_filepath, cols=None):
    file_reader = ReaderCSV(data_filepath)
    input_data = file_reader.read_csv(columns=cols).set_index('id')

    return input_data


def neutralize_pred(strat_dir, model_types):

    model_aggr_fp = strat_dir + '/' + MODEL_AGGREGATION_FILENAME
    full_aggr_dict = Aggregations(model_aggr_fp)
    full_aggr_dict.load()
    aggr_dict = full_aggr_dict.get_models_aggr(model_types)

    predictions_fst_fp = [
        strat_dir + '/' + PREDICTIONS_FILENAME + d_t + PRED_FST_SUFFIX
        for d_t in PREDICTION_TYPES
    ]

    data_types_files = list(zip(PREDICTION_TYPES, predictions_fst_fp))

    for data_type, fpath in data_types_files:

        if data_type != VALID_TYPE:
            continue

        # TODO : if compute, read eras in tournament files
        data_type_eras = get_eras(data_type)
        data_type_pred = load_data(fpath)

        start_time = time.time()
        for aggr, aggr_desc in aggr_dict.items():
            print("")
            print(" ******** aggr: {} ********".format(aggr))

            aggr_pred_name = AGGR_PREFIX + aggr
            n_pred_name = aggr_pred_name + '_n'
            aggr_pred = data_type_pred[['era', aggr_pred_name]]

            fts_n = []
            if 'sel_ft' in aggr_desc.keys():
                fts_n = aggr_dict['optim_neutr']['sel_ft'].split('|')

            aggr_n_full = pd.DataFrame()
            for era in data_type_eras:

                era_input_data = load_h5_eras(TOURNAMENT_STORE_H5_FP, [era])
                era_aggr_pred = aggr_pred.loc[aggr_pred.era == era]

                if len(fts_n) > 0:
                    valid_col = fts_n + ['era', TARGET_LABEL]
                    era_data = pd.concat(
                        [era_input_data[valid_col], era_aggr_pred], axis=1)
                else:
                    era_data = pd.concat([era_input_data, era_aggr_pred],
                                         axis=1)

                aggr_n_full = aggr_n_full.append(era_data[[n_pred_name]])

            aggr_n_full[n_pred_name] = aggr_n_full.groupby('era').apply(
                lambda x: neutralize(x, aggr_pred_name)).values
            aggr_n_full[n_pred_name] = rank_pred(aggr_n_full[n_pred_name])

            data_type_pred = pd.concat([data_type_pred, aggr_n_full], axis=1)
            aggr_desc['neutralized'] = {'name': n_pred_name}

        if data_type == VALID_TYPE:
            with open(model_aggr_fp, 'w') as fp:
                json.dump(aggr_dict, fp, indent=4)

        data_type_test_fp = 'test_' + data_type + '.csv'
        with open(data_type_test_fp, 'w') as f:
            data_type_pred.to_csv(f, index=True)

        print("--- %s seconds ---" % (time.time() - start_time))

    return
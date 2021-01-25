import os
import sys
import json
import numerapi
import pandas as pd
from reader import ReaderCSV
from common import *


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def load_data(data_filepath, data_col=None):
    file_reader = ReaderCSV(data_filepath)
    input_data = file_reader.read_csv(columns=data_col).set_index('id')

    return input_data


def load_predict_file(pred_fp, model_type):
    predict_data = load_data(pred_fp)
    predict_series = predict_data[model_type]
    predict_df = pd.DataFrame({
        'id': predict_series.index,
        PREDICT_LABEL: predict_series.values
    }).set_index('id')

    print("predict_df: ", predict_df)

    return predict_df


def load_orig_id_era_dt():
    data_filepath = TOURNAMENT_DATA_FP
    file_reader = ReaderCSV(data_filepath)
    orig_df = file_reader.read_csv(
        columns=['id', 'era', 'data_type']).set_index('id')

    return orig_df


def upload_results(strat_dir, pred_l):

    pred_l_suffix = pred_l
    orig_data_id_era_dt = load_orig_id_era_dt()

    pred_validation_fp = strat_dir + \
        '/' + PREDICTIONS_FILENAME + VALID_TYPE + '_' + pred_l_suffix + '.csv'
    pred_test_fp = strat_dir + '/' + PREDICTIONS_FILENAME + TEST_TYPE + '_' + pred_l_suffix + '.csv'
    pred_live_fp = strat_dir + '/' + PREDICTIONS_FILENAME + LIVE_TYPE + '_' + pred_l_suffix + '.csv'

    pred_method = 'aggr_pred_11'

    valid_data = load_predict_file(pred_validation_fp, pred_method)
    test_data = load_predict_file(pred_test_fp, pred_method)
    live_data = load_predict_file(pred_live_fp, pred_method)

    pred_data = pd.concat([valid_data, test_data, live_data], axis=0)

    pred_data = pred_data.reindex(orig_data_id_era_dt.index)

    prediction_fp = strat_dir + '/' + NUMERAI_PRED_FILENAME

    with open(prediction_fp, 'w') as fp:
        pred_data.to_csv(fp)

    credentials = load_json('upload_credentials.json')

    print("credentials: ", credentials)

    # provide api tokens
    public_id = 'GVVAJB4FTYC2YKLBY7TINPR4MOZO4PD5'  # credentials['public_id']
    # credentials['secret_key']
    secret_key = 'WOJS5R6CFC2FG2OKPBSOA5S3URH2DO2QH35XAYSLG4LXYL4OSXLNSPFEAZVVHXKM'
    napi = numerapi.NumerAPI(public_id, secret_key)

    # upload predictions
    submission_id = napi.upload_predictions(prediction_fp)
    # check submission status
    napi.submission_status()

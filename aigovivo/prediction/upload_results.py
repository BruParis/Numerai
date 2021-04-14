import os
import sys
import json
import numerapi
import pandas as pd

from ..reader import ReaderCSV
from ..common import *


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def load_data(data_filepath, data_col=None):
    file_reader = ReaderCSV(data_filepath)
    input_data = file_reader.read_csv(columns=data_col).set_index('id')

    return input_data


def load_predict_file(pred_fp, aggr):
    predict_data = load_data(pred_fp)
    print("predict_data: ", predict_data)

    predict_series = predict_data[aggr]
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


def upload_results(strat_dir, pred_l, aggr):

    pred_l_suffix = pred_l
    orig_data_id_era_dt = load_orig_id_era_dt()

    pred_validation_fp = strat_dir + \
        '/' + COMPUTE_PREDICT_PREFIX + VALID_TYPE + '.csv'
    pred_test_fp = strat_dir + '/' + COMPUTE_PREDICT_PREFIX + TEST_TYPE + '.csv'
    pred_live_fp = strat_dir + '/' + COMPUTE_PREDICT_PREFIX + LIVE_TYPE + '.csv'

    valid_data = load_predict_file(pred_validation_fp, aggr)
    test_data = load_predict_file(pred_test_fp, aggr)
    live_data = load_predict_file(pred_live_fp, aggr)

    pred_data = pd.concat([valid_data, test_data, live_data], axis=0)
    pred_data = pred_data.reindex(orig_data_id_era_dt.index)
    print(" -> pred_data: ", pred_data)
    pred_data = pred_data.rank(pct=True)

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
    model_id = napi.get_models()['aigovivo2']
    print("model_id: ", model_id)
    submission_id = napi.upload_predictions(prediction_fp, model_id=model_id)
    # check submission status
    napi.submission_status(model_id=model_id)

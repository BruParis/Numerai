import json

from ..common import *

def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data

def get_eras(data_type):

    if data_type not in [TRAINING_TYPE] + PREDICTION_TYPES:
        print("Wrong data type")
        return

    eras_fp = DATA_DIRNAME+ '/' + ERAS_DT_JSON_FILE

    eras_dict = load_json(eras_fp)

    res = eras_dict[data_type]

    return res

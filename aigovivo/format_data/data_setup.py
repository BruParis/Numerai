import json
import time

from ..common import *
from ..reader import ReaderCSV

from .h5_data import store_h5

def load_eras(data_fp, data_type=None):
    reader = ReaderCSV(data_fp)

    eras_df = reader.read_csv(
        columns=['era']) if data_type is None else reader.read_csv_matching(
            'data_type', [data_type], columns=['era'])
    print("eras_df: ", eras_df.era)
    eras_l = list(set(eras_df.era.values))
    if len(eras_l) > 1:
        eras_l = sorted(eras_l, key=lambda x: int(x.replace('era', '')))

    return eras_l


def extract_era_data_type():
    eras_dict = dict()

    # First training
    if not COMPUTE_BOOL:
        eras_dict[TRAINING_TYPE] = load_eras(TRAINING_DATA_FP)

    # Then tournament training
    for data_t in PREDICTION_TYPES:

        data_type_eras = load_eras(TOURNAMENT_DATA_FP, data_t)
        eras_dict[data_t] = data_type_eras

    eras_fp = DATA_DIRNAME + '/' + ERAS_DT_JSON_FILE

    with open(eras_fp, 'w') as fp:
        json.dump(eras_dict, fp, indent=4)

    return


def data_setup():
    start_time = time.time()

    if not COMPUTE_BOOL:
        store_h5()

    extract_era_data_type()

    print("--- %s seconds ---" % (time.time() - start_time))

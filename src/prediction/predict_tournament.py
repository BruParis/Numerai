import os
import json
import pandas as pd
import time

from common import *
from reader import ReaderCSV, load_h5_eras
from models import ModelType, ModelConstitution
from prediction import PredictionOperator

ERA_BATCH_SIZE = 32
#ERA_BATCH_SIZE = 2


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


def make_prediction_fst(strat, strat_dir, model_dict, data_types_fp, eras_type_df, model_types):
    pred_descr = {}
    file_w_header = {d_t: True for d_t, _ in data_types_fp}

    for data_t, fpath in data_types_fp:

        eras_df = eras_type_df.loc[eras_type_df['data_type'] == data_t]
        eras_list = eras_df['era'].drop_duplicates().values

        eras_batches = list_chunks(
            eras_list) if data_t is not VALID_TYPE else [eras_list]

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

            pred_op = PredictionOperator(
                strat, strat_dir, model_dict, model_types, data_t, bMultiProcess=False)

            pred_data = pred_op.make_fst_layer_predict(input_type_data)

            # stitch eras back
            pred_data = pd.concat(
                [pred_data, input_type_data['era']], axis=1)

            if file_w_header[data_t]:
                pred_descr[data_t] = pred_data.columns.values.tolist()

            write_mode = 'w' if file_w_header[data_t] else 'a'
            with open(fpath, write_mode) as f:
                pred_data.to_csv(
                    f, header=file_w_header[data_t], index=True)
                file_w_header[data_t] = False

    return pred_descr


def make_prediction_snd(strat, strat_dir, model_dict, data_types_fp, model_types):
    pred_descr = {}

    for data_type, fst_layer_path, snd_layer_path in data_types_fp:

        if not os.path.exists(fst_layer_path):
            print("fst layer file not found at:", fst_layer_path)
            continue

        fst_layer_data = load_data(fst_layer_path)

        pred_op = PredictionOperator(
            strat, strat_dir, model_dict, model_types, data_type, bMultiProcess=True)

        snd_layer_pred_data = pred_op.make_full_predict(fst_layer_data)

        # stitch eras back
        snd_layer_pred_data = pd.concat(
            [snd_layer_pred_data, fst_layer_data['era']], axis=1)

        pred_descr[data_type] = snd_layer_pred_data.columns.values.tolist()

        with open(snd_layer_path, 'w') as f:
            snd_layer_pred_data.to_csv(f, index=True)

    return pred_descr


def make_prediction(strat_dir, strat, layer, data_types):

    print('data_types: ', data_types)

    model_dict_fp = strat_dir + '/' + MODEL_CONSTITUTION_FILENAME
    model_dict = load_json(model_dict_fp)

    if 'predictions' not in model_dict.keys():
        model_dict['predictions'] = dict()

    # Fst layer
    predictions_fst_fp = [strat_dir + '/' + PREDICTIONS_FILENAME +
                          d_t + PRED_FST_SUFFIX for d_t in data_types]
    data_types_fst_fp = list(zip(data_types, predictions_fst_fp))

    start_time = time.time()
    eras_type_df = load_eras_data_type()
    print("--- %s seconds ---" % (time.time() - start_time))
    #model_types_fst = [ModelType.NeuralNetwork]
    model_types_fst = [ModelType.XGBoost, ModelType.RandomForest,
                       ModelType.NeuralNetwork]

    if layer == 'fst':
        pred_descr = make_prediction_fst(
            strat, strat_dir, model_dict, data_types_fst_fp, eras_type_df, model_types_fst)
        model_dict['predictions']['fst_layer'] = pred_descr

    if layer == 'snd':
        # Snd layer
        pred_snd_fp = [strat_dir + '/' + PREDICTIONS_FILENAME +
                       d_t + PRED_SND_SUFFIX for d_t in data_types]
        data_types_snd_fp = list(
            zip(data_types, predictions_fst_fp, pred_snd_fp))

        model_types_snd = [ModelType.XGBoost, ModelType.RandomForest,
                           ModelType.NeuralNetwork]

        pred_descr = make_prediction_snd(
            strat, strat_dir, model_dict, data_types_snd_fp, model_types_snd)

        model_dict['predictions']['snd_layer'] = pred_descr

    print("--- %s seconds ---" % (time.time() - start_time))

    with open(model_dict_fp, 'w') as fp:
        json.dump(model_dict, fp, indent=4)

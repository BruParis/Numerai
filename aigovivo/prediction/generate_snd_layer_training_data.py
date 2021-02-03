import os
import errno
import json
import pandas as pd

from ..reader import ReaderCSV
from ..models import Model, ModelType
from ..data_analysis import proba_to_target_label, rank_proba_models
from ..common import *

from .prediction_operator import PredictionOperator

def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def load_data(data_fp, bValid=True):
    f_r = ReaderCSV(data_fp)
    data_df = f_r.read_csv_matching('data_type', [VALID_TYPE]).set_index(
        "id") if bValid else f_r.read_csv().set_index("id")

    return data_df


def snd_layer_training_data(strat):

    cl_dirname = ERA_CL_DIRNAME if strat == "cluster" else ERA_GRAPH_DIRNAME
    snd_layer_dirpath = cl_dirname + '/' + SND_LAYER_DIRNAME

    try:
        os.makedirs(snd_layer_dirpath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Error with : make dir ", snd_layer_dirpath)
            exit(1)

    model_dict_fp = cl_dirname + '/' + MODEL_CONSTITUTION_FILENAME
    model_dict = load_json(model_dict_fp)

    pred_train_fp = snd_layer_dirpath + '/' + PRED_TRAIN_FILENAME
    pred_valid_fp = snd_layer_dirpath + '/' + PREDICTIONS_FILENAME + VALID_TYPE + PRED_FST_SUFFIX
    input_data_fp = [TRAINING_DATA_FP, TOURNAMENT_DATA_FP]
    data_inoutput_fp = list(
        zip(input_data_fp, [pred_train_fp, pred_valid_fp], [False, True]))

    for data_fp, output_fp, b_valid_t in data_inoutput_fp:

        snd_layer_data = load_data(data_fp, bValid=b_valid_t)

        fst_layer_model_types = [
            ModelType.XGBoost, ModelType.RandomForest, ModelType.NeuralNetwork
        ]

        pred_op_fst_layer = PredictionOperator(strat,
                                               cl_dirname,
                                               model_dict,
                                               fst_layer_model_types,
                                               bMultiProc=False)

        model_dict[SND_LAYER] = dict()
        model_dict[SND_LAYER]['input_models'] = [
            eModel.name for eModel in fst_layer_model_types
        ]

        pred_full_data = snd_layer_data['era']
        for cl in model_dict['clusters'].keys():
            # CHOICE
            # predict snd layer training data with fst layer as whole era
            cl_proba = pred_op_fst_layer.make_cl_predict(snd_layer_data, cl)
            cl_rank = rank_proba_models(cl_proba, fst_layer_model_types)

            #cl_rank.columns = cl_rank_col
            cl_rank = cl_rank.drop('era', axis=1)
            cl_rank = cl_rank.rename(columns=lambda col: cl + '_' + col)
            pred_full_data = pd.concat([pred_full_data, cl_rank], axis=1)

            #rank_cl = rank_proba(proba_cl, model_types_fst)

            #rank_cl.columns = [cl + '_' + col for col in rank_cl.columns]
            #pred_full_data = pd.concat([pred_full_data, rank_cl], axis=1)
        print("pred_full_data: ", pred_full_data)

        pred_full_data = pd.concat(
            [pred_full_data, snd_layer_data[TARGET_LABEL]], axis=1)

        with open(output_fp, 'w') as f:
            pred_full_data.to_csv(f)

    with open(model_dict_fp, 'w') as fp:
        json.dump(model_dict, fp, indent=4)

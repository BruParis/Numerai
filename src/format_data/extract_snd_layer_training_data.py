import os
import errno
import json
import pandas as pd
from reader import ReaderCSV

from models import Model, ModelType
from prediction import PredictionOperator, rank_proba
from common import *


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def load_data():
    file_reader = ReaderCSV(TRAINING_DATA_FP)
    data_df = file_reader.read_csv().set_index("id")

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

    snd_layer_data = load_data()

    print("snd_layer_data: ", snd_layer_data)

    fst_layer_model_types = [ModelType.XGBoost,
                             ModelType.RandomForest, ModelType.NeuralNetwork]

    pred_op_fst_layer = PredictionOperator(
        strat, cl_dirname, model_dict, fst_layer_model_types, TRAINING_TYPE, bMultiProcess=False)

    model_dict[SND_LAYER] = dict()
    model_types_fst = [ModelType.XGBoost, ModelType.RandomForest,
                       ModelType.NeuralNetwork]
    model_dict[SND_LAYER]['input_models'] = [
        eModel.name for eModel in model_types_fst]

    pred_full_data = pd.DataFrame()
    for cl in model_dict['clusters'].keys():
        # CHOICE
        # predict snd layer training data with fst layer as whole era
        proba_cl = pred_op_fst_layer.make_cl_predict(snd_layer_data, cl)

        rank_cl = rank_proba(proba_cl, model_types_fst)

        rank_cl.columns = [cl + '_' + col for col in rank_cl.columns]
        pred_full_data = pd.concat([pred_full_data, rank_cl], axis=1)

    pred_full_data = pd.concat(
        [pred_full_data, snd_layer_data[TARGET_LABEL]], axis=1)
    pred_training_data_fp = cl_dirname + '/' + PRED_TRAIN_FILENAME

    with open(pred_training_data_fp, 'w') as f:
        pred_full_data.to_csv(f)

    with open(model_dict_fp, 'w') as fp:
        json.dump(model_dict, fp, indent=4)

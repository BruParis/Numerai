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


def load_data_matching_id(list_id):
    file_reader = ReaderCSV(TRAINING_DATA_FP)
    data_df = file_reader.read_csv_filter_id(list_id).set_index("id")

    return data_df


def load_data_snd_layer(cl_dirname):
    snd_layer_fp = cl_dirname + '/' + DATA_LAYER_FILENAME
    file_reader = ReaderCSV(snd_layer_fp)
    data_df = file_reader.read_csv().set_index("id")

    print("data_df: ", data_df)
    data_df = data_df[data_df['fst_layer'] == False]

    return data_df


def write_file(filepath, data):

    print("write file: ", filepath)
    with open(filepath, 'w') as f:
        data.to_csv(f)


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

    snd_layer_data_id = load_data_snd_layer(cl_dirname)

    print("snd_layer_data_id.index: ", snd_layer_data_id.index)

    # snd_layer_data = load_data_matching_id(snd_layer_data_id.index)
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
        proba_cl.columns = [cl + '_' + col for col in proba_cl.columns]

        pred_cl = rank_proba(proba_cl, model_types_fst)

        pred_cl.columns = [cl + '_' + col for col in pred_cl.columns]

        pred_cl_full_data = pd.concat([proba_cl, pred_cl], axis=1)

        pred_full_data = pd.concat([pred_full_data, pred_cl_full_data], axis=1)

    trian_pred_data = pred_full_data.iloc[snd_layer_data_id.index]

    # snd layer predict era by era from fst layer predict data
    # pred_fst_layer_data = []
    # eras_list = snd_layer_data['era'].drop_duplicates().values
    # print("eras_df: ", eras_list)
    # for era in eras_list:
    #     print('era: ', era)
    #    era_snd_layer_data = snd_layer_data.loc[snd_layer_data['era'] == era]
    #    era_pred_data = pred_op_fst_layer.make_fst_layer_predict(
    #        era_snd_layer_data)
    #    era_pred_data['era'] = era
    #    pred_fst_layer_data.append(era_pred_data)
    # pred_fst_layer_full_data = pd.concat(pred_fst_layer_data)

    # =================================================================
    # snd layer predict from full fst layer predict data
    # pred_fst_layer_full_data = pred_op_fst_layer.make_fst_layer_predict(
    #     snd_layer_data)
    # pred_fst_layer_full_data = pd.concat(
    #     [snd_layer_data['era'], pred_fst_layer_full_data], axis=1)
    # =================================================================

    snd_layer_training_data_filepath = snd_layer_dirpath + '/' + SND_LAYER_FILENAME
    write_file(snd_layer_training_data_filepath, trian_pred_data)

    with open(model_dict_fp, 'w') as fp:
        json.dump(model_dict, fp, indent=4)

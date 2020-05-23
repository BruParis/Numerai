import os
import sys
import numpy as np
import pandas as pd
import itertools

from pool_map import pool_map

from reader_csv import ReaderCSV
from model_abstract import Model, ModelType
from random_forest_model import RFModel
from xgboost_model import XGBModel
from neural_net_model import NeuralNetwork
from k_nn_model import K_NNModel

# Random Forest
# FST ESTIMATION FOR PARAMTERS BOUNDS :
# n_est : < 268;340
# m_depth :  < 28;32
# min_splt : >= 10
# min_leaf : >= 4


def load_data(data_filename):
    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv().set_index("id")

    return data_df


def generate_model(dirname, eModel, train_data, test_data, model_params, model_debug=False):

    if eModel == ModelType.RandomForest:
        RF_model = RFModel(dirname, train_data,
                           test_data, model_params, model_debug)
        RF_model.build_model()
        return RF_model

    if eModel == ModelType.XGBoost:
        XGB_model = XGBModel(dirname, train_data,
                             test_data, model_params, model_debug)
        XGB_model.build_model()
        return XGB_model

    if eModel == ModelType.NeuralNetwork:
        NeuralNet_model = NeuralNetwork(
            dirname, train_data, test_data, model_params, model_debug)
        NeuralNet_model.build_model()
        return NeuralNet_model

    if eModel == ModelType.K_NN:
        K_NN_model = K_NNModel(dirname, train_data,
                               test_data, model_params, model_debug)
        K_NN_model.build_model()
        return K_NN_model


def init_metrics(dirname, model_type, model_prefix=None):
    column_metrics = Model.get_metrics_labels(model_type)
    filename = Model.get_metrics_filename(model_type, model_prefix)

    metrics_filename = dirname + '/' + filename
    metrics_df = pd.DataFrame(columns=column_metrics)

    with open(metrics_filename, 'w') as f:
        metrics_df.to_csv(f, header=True, index=False)


def append_new_metrics(metrics_filename, new_metrics_row, column_metrics):
    metrics_df = pd.DataFrame(data=new_metrics_row, columns=column_metrics)
    with open(metrics_filename, 'a') as f:
        metrics_df.to_csv(f, header=False, index=False)


def produce_metrics(dirname, model_obj, log_loss, accuracy_score, model_prefix=None):
    filename = Model.get_metrics_filename(model_obj.model_type, model_prefix)
    metrics_filename = dirname + '/' + filename

    column_metrics = Model.get_metrics_labels(model_obj.model_type)

    if model_obj.model_type == ModelType.RandomForest:
        rfm = model_obj.model
        new_metrics_row = np.array([[rfm.n_estimators, rfm.max_features, rfm.max_depth,
                                     rfm.min_samples_split, rfm.min_samples_leaf, log_loss, accuracy_score]])
        append_new_metrics(metrics_filename, new_metrics_row, column_metrics)

    if model_obj.model_type == ModelType.XGBoost:
        xgb = model_obj.model
        new_metrics_row = np.array(
            [[xgb.n_estimators, xgb.max_depth, xgb.learning_rate, log_loss, accuracy_score]])
        append_new_metrics(metrics_filename, new_metrics_row, column_metrics)

    if model_obj.model_type == ModelType.NeuralNetwork:
        neuralnet_params = model_obj.model_params
        new_metrics_row = np.array(
            [[neuralnet_params['num_layers'], neuralnet_params['size_factor'], log_loss, accuracy_score]])
        append_new_metrics(metrics_filename, new_metrics_row, column_metrics)

    if model_obj.model_type == ModelType.K_NN:
        k_nn_params = model_obj.model_params
        new_metrics_row = np.array(
            [[k_nn_params['n_neighbors'], k_nn_params['leaf_size'], k_nn_params['minkowski_dist'], log_loss, accuracy_score]])
        append_new_metrics(metrics_filename, new_metrics_row, column_metrics)


def make_model_params(eModel, model_prefix=None):
    if eModel == ModelType.RandomForest:
        n_est = [220]  # np.linspace(start=180, stop=250, num=20)
        max_d = [10]  # np.linspace(10, 20, num=5)
        model_params_array = map(lambda x: {'n_estimators': int(
            x[0]), 'max_depth': int(x[1])}, itertools.product(*[n_est, max_d]))

        return model_params_array

    if eModel == ModelType.XGBoost:
        n_est = [220]  # np.linspace(start=180, stop=250, num=10)
        max_d = [10]  # np.linspace(10, 20, num=5)
        # eta = learning_rate
        eta = [1.0]  # np.logspace(start=(-1.0), stop=0.0, base=10.0, num=5)

        model_params_array = map(lambda x: {'n_estimators': int(x[0]), 'max_depth': int(
            x[1]), 'learning_rate': float(x[2])}, itertools.product(*[n_est, max_d, eta]))

        return model_params_array

    if eModel == ModelType.NeuralNetwork:
        num_layers = [1]  # np.linspace(start=1, stop=4, num=1)
        layer_size_factor = [0.66]
        model_params_array = map(
            lambda x: {'num_layers': int(x[0]), 'size_factor': float(
                x[1])}, itertools.product(*[num_layers, layer_size_factor]))

        return model_params_array

    if eModel == ModelType.K_NN:
        leaf_size = [20]  # np.linspace(start=20, stop=50, num=1)
        minkowski_dist = [1]  # [1, 2, 3, 4]
        model_params_array = map(
            lambda x: {'n_neighbors': int(model_prefix), 'leaf_size': int(x[0]), 'minkowski_dist': int(
                x[1])}, itertools.product(*[leaf_size, minkowski_dist]))

        return model_params_array


def subset_model_build(sub_dir_path, bMetrics=False, model_debug=False):

    train_filepath = sub_dir_path + '/' + 'training_data.csv'
    test_filepath = sub_dir_path + '/' + 'test_data.csv'

    train_data = load_data(train_filepath)
    test_data = load_data(test_filepath)

    # bMultiProc = False
    # if bMultiProc:
    # Parallelism only shows a slight speed improvment
    # -> not enough memory for mutliple thread w/ full dataset (?)
    # model_metrics_arg = list(zip(itertools.repeat(sub_dir_path), itertools.repeat(
    #     metrics_filename), itertools.repeat(train_data), itertools.repeat(test_data), model_params_array))
    # pool_map(generate_model, model_metrics_arg)
    # else:

    bSaveModel = True
    # model_types = ModelType
    # model_types = [ModelType.K_NN]
    # model_types = [ModelType.RandomForest]
    model_types = [ModelType.XGBoost, ModelType.RandomForest,
                   ModelType.NeuralNetwork, ModelType.K_NN]

    res = []
    for eModel in model_types:

        model_prefix_l = [5, 10, 30] if eModel == ModelType.K_NN else [None]

        for model_prefix in model_prefix_l:
            model_dict = dict()
            model_dict['type'] = eModel
            model_dict['prefix'] = model_prefix

            model_params_array = make_model_params(eModel, model_prefix)
            model_dict['params'] = model_params_array

            if bMetrics:
                if eModel == ModelType.K_NN:
                    init_metrics(sub_dir_path, eModel, model_prefix)
                else:
                    init_metrics(sub_dir_path, eModel)

            best_ll = sys.float_info.max
            for model_params in model_params_array:
                curr_model = generate_model(
                    sub_dir_path, eModel, train_data, test_data, model_params, model_debug=model_debug)
                log_loss, accuracy_score = curr_model.evaluate_model()

                if model_debug:
                    print("log_loss: ", log_loss,
                          " - accuracy_score: ", accuracy_score)

                if bMetrics:
                    if eModel == ModelType.K_NN:
                        produce_metrics(sub_dir_path, curr_model,
                                        log_loss, accuracy_score, model_prefix)
                    else:
                        produce_metrics(sub_dir_path, curr_model,
                                        log_loss, accuracy_score)

                if bSaveModel and (log_loss < best_ll):
                    best_ll = log_loss
                    curr_model.save_model()

                    model_dict['log_loss'] = log_loss
                    model_dict['accuracy_score'] = accuracy_score
                    res.append(model_dict)

    return res


def main():

    # test_data = range(1, 100)
    # print("test_pool - test_data: ", test_data)
    # test_pool(test, test_data)
    # print("pool_map - test_data: ", test_data)
    # pool_map(test, test_data)
    # exit(0)

    dirname = 'data_subsets_036'
    _, subdirs, _ = next(os.walk(dirname))
    print("subdirs: ", subdirs)
    sub_dir_paths = [dirname + '/' + subdirname for subdirname in subdirs]

    bMultiProc = True
    print("sub_dir_paths: ", sub_dir_paths)
    # Seems there is a pb with multiprocess (mult. proc w/ same dataframe?)
    if bMultiProc:
        pool_map(subset_model_build, sub_dir_paths)
        # pool_map(test, sub_dir_paths)
        # test_pool(subset_model_build, sub_dir_paths)
    else:
        for sub_dir_path in sub_dir_paths:
            subset_model_build(sub_dir_path, bMetrics=True, model_debug=False)


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np

from model_abstract import Model, ModelType
from random_forest_model import RFModel
from xgboost_model import XGBModel
from neural_net_model import NeuralNetwork
from k_nn_model import K_NNModel


def build_model(dirname, eModel, train_data, test_data, model_params, model_debug=False):

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


def get_metrics_filename(eModel, model_prefix=None):
    if eModel == ModelType.RandomForest:
        return 'random_forest_metrics.csv'

    if eModel == ModelType.XGBoost:
        return 'xgboost_metrics.csv'

    if eModel == ModelType.NeuralNetwork:
        return 'neural_net_metrics.csv'

    if eModel == ModelType.K_NN:
        filename = str(model_prefix) + '_nn_metrics.csv'
        return filename


def get_metrics_labels(eModel):
    if eModel == ModelType.RandomForest:
        column_metrics = ['n_estimators', 'max_features', 'max_depth',
                          'min_samples_split', 'min_samples_leaf',
                          'log_loss', 'accuracy_score']
        return column_metrics

    if eModel == ModelType.XGBoost:
        column_metrics = ['n_estimators', 'max_depth', 'learning_rate',
                          'log_loss', 'accuracy_score']
        return column_metrics

    if eModel == ModelType.NeuralNetwork:
        column_metrics = ['num_layer', 'size_factor',
                          'log_loss', 'accuracy_score']
        return column_metrics

    if eModel == ModelType.K_NN:
        column_metrics = ['n_neighbors', 'leaf_size',
                          'minkowski_dist', 'log_loss', 'accuracy_score']
        return column_metrics


def init_metrics(dirname, model_type, model_prefix=None):
    column_metrics = get_metrics_labels(model_type)
    filename = get_metrics_filename(model_type, model_prefix)

    metrics_filename = dirname + '/' + filename
    metrics_header = pd.DataFrame(columns=column_metrics)

    return metrics_filename, metrics_header


def produce_metrics(model_obj, log_loss, accuracy_score, model_prefix=None):

    new_metrics_row = []

    if model_obj.model_type == ModelType.RandomForest:
        rfm = model_obj.model
        new_metrics_row = np.array([[rfm.n_estimators, rfm.max_features, rfm.max_depth,
                                     rfm.min_samples_split, rfm.min_samples_leaf, log_loss, accuracy_score]])

    if model_obj.model_type == ModelType.XGBoost:
        xgb = model_obj.model
        new_metrics_row = np.array(
            [[xgb.n_estimators, xgb.max_depth, xgb.learning_rate, log_loss, accuracy_score]])

    if model_obj.model_type == ModelType.NeuralNetwork:
        neuralnet_params = model_obj.model_params
        new_metrics_row = np.array(
            [[neuralnet_params['num_layers'], neuralnet_params['size_factor'], log_loss, accuracy_score]])

    if model_obj.model_type == ModelType.K_NN:
        k_nn_params = model_obj.model_params
        new_metrics_row = np.array(
            [[k_nn_params['n_neighbors'], k_nn_params['leaf_size'], k_nn_params['minkowski_dist'], log_loss, accuracy_score]])

    column_metrics = get_metrics_labels(model_obj.model_type)
    metrics_df = pd.DataFrame(data=new_metrics_row, columns=column_metrics)

    return metrics_df

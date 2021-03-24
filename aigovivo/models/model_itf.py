import pandas as pd
import numpy as np

from .model_abstract import Model, ModelType
from .random_forest_model import RFModel
from .xgboost_model import XGBModel
from .neural_net_model import NeuralNetwork
from .k_nn_model import K_NNModel
from .univ_poly_interpo_model import UnivPolyInterpo


def load_model(subset_dirname, eModel, model_prefix=None):
    if eModel == ModelType.RandomForest:
        RF_model = RFModel(subset_dirname, debug=False)
        RF_model.load_model()
        return RF_model

    if eModel == ModelType.XGBoost:
        XGB_model = XGBModel(subset_dirname, debug=False)
        XGB_model.load_model()
        return XGB_model

    if eModel == ModelType.NeuralNetwork:
        NN_model = NeuralNetwork(subset_dirname, debug=False)
        NN_model.load_model()
        return NN_model

    if eModel == ModelType.K_NN:
        model_params = {"n_neighbors": model_prefix}
        k_nn_model = K_NNModel(subset_dirname,
                               model_params=model_params,
                               debug=False)
        k_nn_model.load_model()
        return k_nn_model

    if eModel == ModelType.UnivPolyInterpo:
        univ_polyinterpo = UnivPolyInterpo(subset_dirname,
                                           model_params=model_params,
                                           debug=False)
        univ_polyinterpo.load_model()
        return univ_polyinterpo


def construct_model(dirname, eModel, model_params, model_debug=False):

    if eModel == ModelType.RandomForest:
        RF_model = RFModel(dirname, model_params, model_debug)
        return RF_model

    if eModel == ModelType.XGBoost:
        XGB_model = XGBModel(dirname, model_params, model_debug)
        return XGB_model

    if eModel == ModelType.NeuralNetwork:
        NeuralNet_model = NeuralNetwork(dirname, model_params, model_debug)
        return NeuralNet_model

    if eModel == ModelType.K_NN:
        K_NN_model = K_NNModel(dirname, model_params, model_debug)
        return K_NN_model

    if eModel == ModelType.UnivPolyInterpo:
        univ_polyinterpo = UnivPolyInterpo(dirname, model_params, model_debug)
        return univ_polyinterpo


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

    if eModel == ModelType.UnivPolyInterpo:
        filename = str(model_prefix) + '_metrics.csv'
        return filename


def get_desc_filename(eModel, model_prefix=None):
    if eModel == ModelType.RandomForest:
        return 'rf_desc.json'

    if eModel == ModelType.XGBoost:
        return 'xgb_desc.json'

    if eModel == ModelType.NeuralNetwork:
        return 'nn_desc.json'

    if eModel == ModelType.K_NN:
        filename = str(model_prefix) + '_nn_desc.json'
        return filename

    if eModel == ModelType.UnivPolyInterpo:
        filename = str(model_prefix) + '_univpolyinterpo.csv'
        return filename
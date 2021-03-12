import pickle
import json
import pandas as pd
import sklearn.metrics as sm

from abc import ABC, abstractmethod
from ..data_analysis import rank_proba
from ..common import *


class Model(ABC):
    def _get_model_name(self, model_type, model_params):
        if model_type == ModelType.K_NN:
            filename = str(model_params['n_neighbors']) + '_nn'
            return filename

        if model_type == ModelType.UnivPolyInterpo:
            filename = str(model_params['degree']) + '_interpo'
            return filename

        return model_type.name

    # for model type XGB -> era column removal already done for xgb.DMatrix

    def _load_model_config(self):
        with open(self.config_filepath, 'r') as fp:
            config = json.load(fp)
            self.n_features = config['n_features']
            self.model_params = config['model_params']
            self.training_score = config['training_score']

    def __init__(self,
                 model_type: ModelType,
                 dirname,
                 model_params=None,
                 debug=False,
                 filename=None):

        self.dirname = dirname
        self.debug = debug

        self.model = None
        self.model_type = model_type
        self.model_params = model_params
        self.n_features = 0

        self.model_name = self._get_model_name(self.model_type,
                                               self.model_params)
        model_file_suffix = '_model.sav'

        self.training_score = {'log_loss': None, 'accuracy_score': None}
        self.filepath = self.dirname + '/' + \
            filename if filename is not None else self.dirname + \
            '/' + self.model_name + model_file_suffix
        self.config_filepath = self.dirname + '/' + \
            filename if filename is not None else self.dirname + \
            '/' + self.model_name + '_model_config.json'

    @abstractmethod
    def build_model(self, model_params, random_search=False):
        pass

    @abstractmethod
    def predict(self, data_input):
        pass

    @abstractmethod
    def predict_proba(self, data_input):
        pass

    def write_metrics(self, train_eval, num_metrics):
        return

    def model_config(self):
        config = {'n_features': self.n_features}
        config['model_type'] = self.model_type.name
        config['model_params'] = self.model_params
        config['training_score'] = self.training_score

        return config

    def save_model(self):
        pickle.dump(self.model, open(self.filepath, 'wb'))
        with open(self.config_filepath, 'w') as fp:
            json.dump(self.model_config(), fp, indent=4)

        return self.filepath, self.config_filepath

    def load_model(self):
        self.model = pickle.load(open(self.filepath, 'rb'))
        self._load_model_config()

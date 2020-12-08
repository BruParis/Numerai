import pickle
import enum
import json
import pandas as pd
import sklearn.metrics as sm

from abc import ABC, abstractmethod
from common import *


class ModelType(enum.Enum):
    RandomForest = 1
    XGBoost = 2
    K_NN = 3
    NeuralNetwork = 4


class Model(ABC):

    def _get_model_name(self, model_type, model_params):
        if model_type == ModelType.K_NN:
            filename = str(model_params['n_neighbors']) + '_nn'
            return filename

        return model_type.name

    # for model type XGB -> era column removal already done for xgb.DMatrix

    def _format_input_target(self, data_df):

        # if self.model_type is not ModelType.XGBoost:
        if ERA_LABEL in data_df.columns:
            data_df = data_df.drop([ERA_LABEL], axis=1)

        input_df = data_df.drop([TARGET_LABEL], axis=1)
        target_df = data_df.loc[:, [TARGET_LABEL]]

        # mutiply target value to get target class index
        target_df = TARGET_FACT_NUMERIC * target_df

        return input_df, target_df

    def _display_cross_tab(self, test_input, test_target):

        cross_tab = pd.crosstab(test_target[TARGET_LABEL], test_target['prediction'],
                                rownames=['Actual Target'],
                                colnames=['Predicted Target'])
        cross_tab_perc = cross_tab.apply(lambda r: (
            100*round(r/r.sum(), 2)).astype(int), axis=1)
        cross_tab['Total'] = cross_tab.sum(axis=1)
        cross_tab = cross_tab.append(cross_tab.sum(), ignore_index=True)

        print("cross_tab: ", cross_tab)
        cross_tab_perc['Total'] = cross_tab['Total']
        cross_tab_perc = cross_tab_perc.append(
            cross_tab.iloc[-1], ignore_index=True)
        print("Cross Tab %: ", cross_tab_perc)

    def _load_model_config(self):
        with open(self.config_filepath, 'r') as fp:
            config = json.load(fp)
            self.n_features = config['n_features']
            self.model_params = config['model_params']
            self.training_score = config['training_score']

    def __init__(self, model_type: ModelType, dirname, model_params=None,
                 debug=False, filename=None):

        self.dirname = dirname
        self.debug = debug

        self.model = None
        self.model_type = model_type
        self.model_params = model_params
        self.n_features = 0

        self.model_name = self._get_model_name(
            self.model_type, self.model_params)
        model_file_suffix = '_model.sav'

        self.training_score = {'log_loss': None, 'accuracy_score': None}
        self.filepath = self.dirname + '/' + \
            filename if filename is not None else self.dirname + \
            '/' + self.model_name + model_file_suffix
        self.config_filepath = self.dirname + '/' + \
            filename if filename is not None else self.dirname + \
            '/' + self.model_name + '_model_config.json'

    @abstractmethod
    def build_model(self, model_params):
        pass

    @abstractmethod
    def predict(self, data_input):
        pass

    @abstractmethod
    def predict_proba(self, data_input):
        pass

    # Neural Net, XGBoost got their own implementations
    def evaluate_model(self, test_data):
        if self.model is None:
            print("Model not yet built!")
            return

        if test_data is None:
            print("No test data provided!")
            return

        test_input, test_target = self._format_input_target(test_data)
        test_target['prediction'] = self.predict(test_input)

        # test_fst_rows = test_target[0:10]
        # pred_proba = clf.predict_proba(test_input)[0:10].round(2)
        # test_fst_rows['proba'] = pred_proba.tolist()
        # print("test_fst_rows: ", test_fst_rows)

        if self.debug:
            self._display_cross_tab(test_input, test_target)

        test_proba = self.predict_proba(test_input)

        if self.debug:
            print("test proba: ", test_proba)

        test_target['proba'] = test_proba.tolist()
        log_loss = sm.log_loss(test_target[TARGET_LABEL], test_proba.tolist())
        accuracy_score = sm.accuracy_score(
            test_target[TARGET_LABEL], test_target['prediction'])

        self.training_score['log_loss'] = log_loss
        self.training_score['accuracy_score'] = accuracy_score

        if self.debug:
            print("model evaluation - log_loss: ", log_loss,
                  " - accuracy_score: ", accuracy_score)

        return log_loss, accuracy_score

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

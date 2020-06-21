import json
import math
import model_itf
import itertools
import numpy as np
import pandas as pd

from model_abstract import ModelType
from pool_map import pool_map
from reader_csv import ReaderCSV

TARGET_LABEL = 'target_kazutsugi'
COL_PROBA_NAMES = ['proba_0.0', 'proba_0.25',
                   'proba_0.5', 'proba_0.75', 'proba_1.0']


class PredictionOperator():

    def _load_corr_subset(self, subset_name):
        corr_path = self.dirname + '/' + subset_name + '/eras_corr.csv'
        corr = ReaderCSV(corr_path).read_csv().set_index('corr_features')
        return corr

    def _load_models_json(self):
        with open(self.layer_distrib_filepath, 'r') as f:
            layer_distrib = json.load(f)

            return layer_distrib

    def _load_subset_models_json(self):
        with open(self.layer_distrib_filepath, 'r') as f:
            layer_distrib = json.load(f)

            for sub_name, subset_desc in layer_distrib["subsets"].items():
                subset_desc['corr_mat'] = self._load_corr_subset(sub_name)

            return layer_distrib

    def _model_proba(self, model_path, model_type, model_prefix, input_data):
        print("path: ", model_path)
        print("model_type: ", model_type)
        print("model_prefix: ", model_prefix)
        model = model_itf.load_model(model_path, model_type, model_prefix)

        prediction_proba = model.predict_proba(input_data)

        columns_labels = [model.model_name + '_' +
                          proba_label for proba_label in COL_PROBA_NAMES]

        prediction_proba_df = pd.DataFrame(
            prediction_proba, input_data.index, columns=columns_labels)

        return prediction_proba_df

    def _aggregated_proba(self, input_data_ft, sub_name, subset):
        sub_path = self.dirname + '/' + sub_name

        models_proba_full = pd.DataFrame(dtype=np.float32)
        for col in models_proba_full.columns:
            models_proba_full[col].values[:] = 0.0

        all_model_proba = []
        for model_desc in subset['models']:
            eModel = ModelType[model_desc['type']]

            if not (eModel in self.model_types):
                continue

            prefix = model_desc['prefix']

            pred_proba = self._model_proba(
                sub_path, eModel, prefix, input_data_ft)

            all_model_proba.append(pred_proba)

        models_proba_full = pd.concat(all_model_proba, axis=1)

        return models_proba_full

    def _make_sub_proba(self, input_data, input_data_corr, sub_name, subset):

        subset_ft = subset['features']
        input_data_ft = input_data[subset_ft]

        sub_proba = self._aggregated_proba(input_data_ft, sub_name, subset)

        sub_ft = subset['features']
        sub_ft_corr = input_data_corr.loc[input_data_corr.index.isin(
            sub_ft)][sub_ft]

        # ==========================
        # Need to check best formula for era (sub)datasets similarity
        # / len (sub_ft) or len(sub_ft) ** 2 ?
        sub_corr_diff = (sub_ft_corr - subset['corr_mat'].values)
        sub_corr_dist = (sub_corr_diff / len(sub_ft)) ** 2

        sub_simi = math.sqrt(sub_corr_dist.values.sum())
        # ==========================

        sub_simi_proba = {}
        sub_simi_proba['simi'] = sub_simi
        sub_simi_proba['proba'] = sub_proba

        return sub_simi_proba

    def __init__(self, dirname, layer_distrib_file, model_types, bMultiProcess=False):
        self.dirname = dirname
        self.layer_distrib_filepath = self.dirname + '/' + layer_distrib_file

        self.model_types = model_types
        self.model_prefixes = model_types
        self.bMultiProcess = bMultiProcess

    def make_fst_layer_predict(self, input_data):
        self.layer_distrib = self._load_subset_models_json()

        input_data_corr = input_data.corr()

        subsets = self.layer_distrib['subsets']

        if self.bMultiProcess:
            sub_proba_param = list(zip(itertools.repeat(
                input_data), itertools.repeat(input_data_corr), subsets.keys(), subsets.values()))
            sub_proba_l = pool_map(self._make_sub_proba,
                                   sub_proba_param, collect=True, arg_tuple=True)
            sub_proba = dict(zip(subsets.keys(), sub_proba_l))
        else:
            sub_proba = {sub_name: self._make_sub_proba(input_data, input_data_corr, sub_name, subset)
                         for sub_name, subset in subsets.items()}

        total_sim = sum(sub_proba['simi']
                        for _, sub_proba in sub_proba.items())
        full_proba = sum(sub_proba['proba'] * sub_proba['simi']
                         for _, sub_proba in sub_proba.items()) / total_sim

        return full_proba

    def make_full_predict(self, data_df):

        if 'era' in data_df.columns:
            data_df = data_df.drop('era', axis=1)

        if 'data_type' in data_df.columns:
            data_df = data_df.drop('data_type', axis=1)

        if TARGET_LABEL in data_df.columns:
            data_df = data_df.drop(TARGET_LABEL, axis=1)

        self.layer_distrib = self._load_models_json()

        print("self.layer_distrib: ", self.layer_distrib)
        model_desc_l = self.layer_distrib['models']

        all_model_proba = []
        for model_desc in model_desc_l:

            eModel = ModelType[model_desc['type']]

            if not (eModel in self.model_types):
                continue

            prefix = model_desc['prefix']

            pred_model_proba = self._model_proba(
                self.dirname, eModel, prefix, data_df)

            print("pred_model_proba: ", pred_model_proba)

            all_model_proba.append(pred_model_proba)

        full_proba = pd.concat(all_model_proba, axis=1)

        return full_proba

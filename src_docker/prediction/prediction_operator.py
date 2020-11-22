import json
import math
import os
import itertools
import numpy as np
import pandas as pd

from models import ModelType, load_model
from threadpool import pool_map
from reader import ReaderCSV
from common import *


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

            self.layer_distrib = layer_distrib
            self.clusters = self.layer_distrib['clusters']
            for sub_name, subset_desc in self.clusters.items():
                subset_desc['corr_mat'] = self._load_corr_subset(sub_name)

            return layer_distrib

    def _model_proba(self, model_path, model_type, model_prefix, input_data):
        print("path: ", model_path)
        print("model_type: ", model_type)
        print("model_prefix: ", model_prefix)
        model_dirname = os.path.dirname(model_path)
        model = load_model(model_dirname, model_type, model_prefix)

        prediction_proba = model.predict_proba(input_data)

        columns_labels = [model.model_name + '_' +
                          proba_label for proba_label in COL_PROBA_NAMES]

        prediction_proba_df = pd.DataFrame(
            prediction_proba, input_data.index, columns=columns_labels)

        return prediction_proba_df

    def _aggregated_proba(self, input_data_ft, subset):

        models_proba_full = pd.DataFrame(dtype=np.float32)
        for col in models_proba_full.columns:
            models_proba_full[col].values[:] = 0.0

        all_model_proba = []
        for model_desc in subset['models']:
            eModel = ModelType[model_desc['type']]

            if not (eModel in self.model_types):
                continue

            prefix = model_desc['prefix']
            model_fp = model_desc['model_filepath']

            pred_proba = self._model_proba(
                model_fp, eModel, prefix, input_data_ft)

            all_model_proba.append(pred_proba)

        models_proba_full = pd.concat(all_model_proba, axis=1)

        return models_proba_full

    def _make_cl_proba(self, input_data, input_data_corr, cluster):

        cl_ft = cluster['selected_features']
        input_data_ft = input_data[cl_ft]

        cl_proba = self._aggregated_proba(input_data_ft, cluster)
        print('cl_proba: ', cl_proba)

        cl_ft_corr = input_data_corr.loc[input_data_corr.index.isin(
            cl_ft)][cl_ft]

        cl_score_proba = {}
        cl_score_proba['proba'] = cl_proba

        # if self.strat == STRAT_CLUSTER:
        #     # CHOICE
        #     # Use cl score, or mean of target_corr in cluster as ponderation for proba?
        #     # cl_score_proba['weight'] = cluster['score']
        #     # or same as era_graph ????? -> need testing
        #     cl_score_proba['weight'] = cluster['mean_t_corr']
        # elif self.strat == STRAT_ERA_GRAPH:
        # ==========================
        # CHOICE
        # Need to check best formula for era (sub)datasets similarity
        # / len (sub_ft) or len(sub_ft) ** 2 ?

        cl_corr_diff = (cl_ft_corr - cluster['corr_mat'].values)
        cl_corr_dist = (cl_corr_diff / len(cl_ft)) ** 2

        #print("cl_corr_dist: ", cl_corr_dist)

        cl_score = math.sqrt(cl_corr_dist.values.sum())
        # ==========================

        cl_score_proba['weight'] = cl_score

        return cl_score_proba

    def __init__(self, strat, dirname, layer_distrib_file, model_types, bMultiProcess=False):
        self.strat = strat
        self.dirname = dirname
        self.layer_distrib_filepath = self.dirname + '/' + layer_distrib_file

        self.model_types = model_types
        self.model_prefixes = model_types
        self.bMultiProcess = bMultiProcess

    def make_fst_layer_predict(self, input_data):
        self._load_subset_models_json()

        input_data_corr = input_data.corr()

        cl_keys = self.clusters.keys()
        cl_values = self.clusters.values()
        if self.bMultiProcess:
            sub_proba_param = list(zip(itertools.repeat(
                input_data), itertools.repeat(input_data_corr), cl_values))
            sub_proba_l = pool_map(
                self._make_cl_proba, sub_proba_param, collect=True, arg_tuple=True)
            sub_proba = dict(zip(cl_keys, sub_proba_l))
        else:
            sub_proba = {sub_name: self._make_cl_proba(input_data, input_data_corr, subset)
                         for sub_name, subset in self.clusters.items()}

        print("sub_proba: ", sub_proba)

        total_sim = sum(sub_proba['weight']
                        for _, sub_proba in sub_proba.items())
        full_proba = sum(sub_proba['proba'] * sub_proba['weight']
                         for _, sub_proba in sub_proba.items()) / total_sim

        return full_proba

    def make_full_predict(self, data_df, bSnd=True):

        if 'era' in data_df.columns:
            data_df = data_df.drop('era', axis=1)

        if 'data_type' in data_df.columns:
            data_df = data_df.drop('data_type', axis=1)

        if TARGET_LABEL in data_df.columns:
            data_df = data_df.drop(TARGET_LABEL, axis=1)

        self.layer_distrib = self._load_models_json()

        print("self.layer_distrib: ", self.layer_distrib)
        model_desc_l = self.layer_distrib['snd_layer']['models'] if bSnd else {
        }

        all_model_proba = []
        for model_desc in model_desc_l:

            eModel = ModelType[model_desc['type']]

            if not (eModel in self.model_types):
                continue

            prefix = model_desc['prefix']
            model_fp = model_desc['model_filepath']

            pred_model_proba = self._model_proba(
                model_fp, eModel, prefix, data_df)

            print("pred_model_proba: ", pred_model_proba)

            all_model_proba.append(pred_model_proba)

        full_proba = pd.concat(all_model_proba, axis=1)

        return full_proba

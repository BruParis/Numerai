import math
import os
import itertools
import numpy as np
import pandas as pd

from ..models import ModelType, load_model
from ..threadpool import pool_map
from ..reader import ReaderCSV
from ..common import *


class PredictionOperator():
    def _load_corr_subset(self, subset_name):
        corr_path = self.dirname + '/' + subset_name + '/eras_corr.csv'
        corr = ReaderCSV(corr_path).read_csv().set_index('corr_features')
        return corr

    def _model_predict_proba(self, model_path, model_type, model_prefix,
                             input_data):
        print("path: ", model_path)
        print("model_type: ", model_type)
        print("model_prefix: ", model_prefix)
        model_dirname = os.path.dirname(model_path)
        print("model_path: ", model_path)
        model = load_model(model_dirname, model_type, model_prefix)

        prediction_proba = model.predict_proba(input_data)

        columns_labels = [
            model.model_name + '_' + proba_label
            for proba_label in COL_PROBA_NAMES
        ]

        prediction_proba_df = pd.DataFrame(prediction_proba,
                                           input_data.index,
                                           columns=columns_labels)

        return prediction_proba_df

    def _model_proba(self, input_data, cluster, eModel, model_desc):

        cl_ft = cluster['selected_features']
        input_data_ft = input_data[cl_ft]
        # input_target = input_data[TARGET_LABEL]

        models_proba_full = pd.DataFrame(dtype=np.float32)
        for col in models_proba_full.columns:
            models_proba_full[col].values[:] = 0.0

        prefix = None  #model_desc['prefix']
        model_fp = model_desc['model_filepath']

        pred_proba = self._model_predict_proba(model_fp, eModel, prefix,
                                               input_data_ft)

        return pred_proba

    def _make_cl_proba_dict(self, input_data, cl_dict):

        cl_all_models_proba = {}
        for model, model_desc in cl_dict['models'].items():

            eModel = ModelType[model]

            if not (eModel in self.model_types):
                continue

            cl_model_proba = self._model_proba(input_data, cl_dict, eModel,
                                               model_desc)

            cl_all_models_proba[eModel.name] = cl_model_proba

            # cl_score_proba = {}
            # cl_score_proba['proba'] = cl_model_proba
            # if self.strat == STRAT_CLUSTER:
            #     # CHOICE
            #     # Use cl score, or mean of target_corr in cluster as ponderation for proba?
            #     # cl_score_proba['weight'] = cluster['score']
            #     # or same as era_graph ????? -> need testing
            #     cl_score_proba['weight'] = cluster['mean_t_corr']
            # elif self.strat == STRAT_ERA_GRAPH:
            # ==========================
            # CHOICE
            # cl_ft_corr = input_data_corr.loc[input_data_corr.index.isin(
            #     cl_ft)][cl_ft]
            # Need to check best formula for era (sub)datasets similarity
            # / len (sub_ft) or len(sub_ft) ** 2 ?
            # cl_corr_diff = (cl_ft_corr - cluster['corr_mat'].values)
            # cl_corr_dist = (cl_corr_diff / len(cl_ft)) ** 2
            # print("cl_corr_dist: ", cl_corr_dist)
            # cl_score = math.sqrt(cl_corr_dist.values.sum())
            # cl_score_proba['weight'] = cl_score
            # ==========================

        return cl_all_models_proba

    def __init__(self, dirname, model_const, model_types, bMultiProc=False):
        self.dirname = dirname
        self.model_const = model_const
        self.clusters = self.model_const['clusters']

        self.model_types = model_types
        self.model_prefixes = model_types
        self.bMultiProc = bMultiProc

    def make_cl_predict(self, input_data, cl):
        cl_dict = self.clusters[cl]

        cl_proba = self._make_cl_proba_dict(input_data, cl_dict)

        full_proba = pd.DataFrame()
        for eModel in self.model_types:

            full_proba = pd.concat([full_proba, cl_proba[eModel.name]], axis=1)

        full_proba = pd.concat([input_data['era'], full_proba], axis=1)

        return full_proba

    def make_snd_layer_predict(self, data_df, bSnd=True):

        snd_layer_desc = self.model_const['snd_layer']
        gen_model = snd_layer_desc['models']['gen_models'] if bSnd else {}

        full_proba = pd.DataFrame()
        for eModel in self.model_types:

            if eModel.name not in gen_model:
                continue

            model_desc = gen_model[eModel.name]
            print("model_name: ", eModel.name)
            print("model_desc: ", model_desc)

            prefix = None
            model_fp = model_desc['model_filepath']

            pred_model_proba = self._model_predict_proba(
                model_fp, eModel, prefix, data_df)

            print("pred_model_proba: ", pred_model_proba)
            full_proba = pd.concat([full_proba, pred_model_proba], axis=1)

        return full_proba

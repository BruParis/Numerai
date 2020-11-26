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

    def _model_predict_proba(self, model_path, model_type, model_prefix, input_data):
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

    def _compute_proba_t_corr(self, proba, input_target):
        print("************************************************")
        print("*******                                  *******")
        print("******* COMPUTE PROBA TARGET CORRELATION *******")
        print("*******                                  *******")
        print("************************************************")

        # TODO :
        # Duplicated code from final_prediction
        proba_col_to_classes = dict(
            zip(proba.columns, TARGET_CLASSES))
        proba_target_label = proba.rename(columns=proba_col_to_classes)
        pred_target_label = proba_target_label.idxmax(axis=1).astype(float)

        proba_t_corr = np.corrcoef(pred_target_label, input_target)[0, 1]

        return proba_t_corr

    def _model_proba(self, input_data, cluster, eModel, model_desc):

        cl_ft = cluster['selected_features']
        input_data_ft = input_data[cl_ft]
        input_target = input_data[TARGET_LABEL]

        models_proba_full = pd.DataFrame(dtype=np.float32)
        for col in models_proba_full.columns:
            models_proba_full[col].values[:] = 0.0

        prefix = model_desc['prefix']
        model_fp = model_desc['model_filepath']

        pred_proba = self._model_predict_proba(
            model_fp, eModel, prefix, input_data_ft)

        if self.data_type == VALID_TYPE:
            model_desc['valid_corr'] = self._compute_proba_t_corr(
                pred_proba, input_target)

        return pred_proba

    def _make_cl_proba(self, input_data, input_data_corr, cluster):

        cl_all_models_proba = {}
        for model, model_desc in cluster['models'].items():

            eModel = ModelType[model]

            if not (eModel in self.model_types):
                continue

            cl_model_proba = self._model_proba(
                input_data, cluster, eModel, model_desc)

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

    def __init__(self, strat, dirname, model_const, model_types, data_type,
                 bMultiProcess=False):
        self.strat = strat
        self.data_type = data_type
        self.dirname = dirname
        self.model_const = model_const
        self.clusters = self.model_const['clusters']

        self.model_types = model_types
        self.model_prefixes = model_types
        self.bMultiProcess = bMultiProcess

    def make_fst_layer_predict(self, input_data):

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
            sub_proba = {cl_name: self._make_cl_proba(input_data, input_data_corr, cluster)
                         for cl_name, cluster in self.clusters.items()}

        # very complicated -> need simplification
        full_proba = pd.DataFrame()
        for eModel in self.model_types:

            cl_w = {cl: cl_desc['models'][eModel.name]['valid_corr']
                    for cl, cl_desc in self.clusters.items()
                    if cl_desc['models'][eModel.name]['valid_corr'] > 0}
            total_w = sum([w for cl, w in cl_w.items()])
            full_proba_model = sum(proba[eModel.name] * cl_w[cl]
                                   for cl, proba in sub_proba.items()
                                   if cl in cl_w.keys()) / total_w
            full_proba = pd.concat([full_proba, full_proba_model], axis=1)

        return full_proba

    def make_full_predict(self, data_df, bSnd=True):

        if 'era' in data_df.columns:
            data_df = data_df.drop('era', axis=1)

        if 'data_type' in data_df.columns:
            data_df = data_df.drop('data_type', axis=1)

        if TARGET_LABEL in data_df.columns:
            data_df = data_df.drop(TARGET_LABEL, axis=1)

        model_desc_l = self.model_const['snd_layer']['models'] if bSnd else {}

        all_model_proba = []
        for model_desc in model_desc_l:

            eModel = ModelType[model_desc['type']]

            if not (eModel in self.model_types):
                continue

            prefix = model_desc['prefix']
            model_fp = model_desc['model_filepath']

            # pred_model_proba = self._model_proba(
            #     model_fp, eModel, prefix, data_df)
            pred_model_proba = self._model_predict_proba(
                model_fp, eModel, prefix, data_df)

            print("pred_model_proba: ", pred_model_proba)

            all_model_proba.append(pred_model_proba)

        full_proba = pd.concat(all_model_proba, axis=1)

        return full_proba

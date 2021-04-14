import math
import os
import itertools
import numpy as np
import pandas as pd

from ..models import load_model, ModelDescription
from ..threadpool import pool_map
from ..reader import ReaderCSV
from ..common import *
from ..data_analysis import rank_proba_models


def model_predic_rank(eModel, model, model_fts, input_data):
    columns_labels = [
        eModel.name + '_' + proba_label for proba_label in COL_PROBA_NAMES
    ]

    proba_df = model.predict_proba(input_data[model_fts])
    proba_df = pd.DataFrame(proba_df, input_data.index, columns=columns_labels)

    rank_df = rank_proba_models(proba_df, [eModel])

    return rank_df


class PredictionOperator():
    def _load_corr_subset(self, subset_name):
        corr_path = self.dirname + '/' + subset_name + '/eras_corr.csv'
        corr = ReaderCSV(corr_path).read_csv().set_index('corr_features')
        return corr

    def _make_cl_proba_dict(self, input_data, cl_dict):

        cl_all_models_proba = {}
        for model, model_d_fp in cl_dict['models'].items():

            eModel = ModelType[model]

            if not (eModel in self.model_types):
                continue

            model_desc = ModelDescription('fst', model_d_fp, eModel)
            model_desc.load()
            model_fp = model_desc.model_fp

            cl_ft = cl_dict['selected_features'].split('|')
            input_data_ft = input_data[cl_ft]

            prefix = None  #model_desc['prefix']

            model_dirname = os.path.dirname(model_fp)
            model = load_model(model_dirname, eModel, prefix)

            cl_model_proba = model.predict_proba(input_data_ft)

            cl_all_models_proba[eModel.name] = cl_model_proba

        return cl_all_models_proba

    def __init__(self, dirname, strat_c, model_types, bMultiProc=False):
        self.dirname = dirname
        self.strat_c = strat_c
        self.clusters = self.strat_c.clusters

        self.model_types = model_types
        self.model_prefixes = model_types
        self.bMultiProc = bMultiProc

    def make_cl_predict(self, input_data, cl):
        cl_dict = self.clusters[cl]

        cl_proba = self._make_cl_proba_dict(input_data, cl_dict)

        full_proba = pd.DataFrame()
        for eModel in self.model_types:

            columns_labels = [
                eModel.name + '_' + proba_label
                for proba_label in COL_PROBA_NAMES
            ]

            cl_proba_df = pd.DataFrame(cl_proba[eModel.name],
                                       input_data.index,
                                       columns=columns_labels)

            full_proba = pd.concat([full_proba, cl_proba_df], axis=1)

        full_proba = pd.concat([input_data['era'], full_proba], axis=1)

        return full_proba

    def make_snd_layer_predict(self, data_df, bSnd=True):

        return None

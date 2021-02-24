import os
import sys
import numpy as np
import pandas as pd
import itertools
import sklearn.metrics as sm

from ..common import *
from ..data_analysis import rank_proba, valid_score
from .model_abstract import Model, ModelType
from .model_itf import init_metrics, produce_metrics, generate_model


class ModelGenerator():
    def _init_metrics(self):
        metrics_filepath, metrics_header = init_metrics(
            self.dir_path, self.model_type, self.model_prefix)

        with open(metrics_filepath, 'w') as f:
            metrics_header.to_csv(f, header=True, index=False)
            self.metrics_filepath = metrics_filepath

    def _append_new_metrics(self, model, log_loss, accuracy_score):

        if self.model_type is None:
            print("Error: File path absent to append new metrics.")
            return None

        metrics_df = produce_metrics(model, log_loss, accuracy_score,
                                     self.model_prefix)

        with open(self.metrics_filepath, 'a') as f:
            metrics_df.to_csv(f, header=False, index=False)

    def _format_input_target(self, data_df):

        data_aux = data_df.copy()

        # if self.model_type is not ModelType.XGBoost:
        if ERA_LABEL in data_aux.columns:
            data_aux = data_aux.drop([ERA_LABEL], axis=1)

        input_df = data_aux.drop([TARGET_LABEL], axis=1)
        target_df = data_aux.loc[:, [TARGET_LABEL]]

        # mutiply target value to get target class index
        target_df = TARGET_FACT_NUMERIC * target_df

        return input_df, target_df

    def _oversampling(self, train_data):
        # TODO : balance classes
        sampling_dict = {
            t: len(train_data.loc[train_data[TARGET_LABEL] == t].index)
            for t in TARGET_VALUES
        }
        print("sampling_dict: ", sampling_dict)
        max_class = max(sampling_dict, key=sampling_dict.get)
        print("max_class: ", max_class)
        supp_fact_dict = {
            k: (sampling_dict[max_class] - sampling_dict[k]) / sampling_dict[k]
            for k, v in sampling_dict.items()
        }
        print("supp_fact_dict: ", supp_fact_dict)

        supp_train_data = pd.DataFrame()
        for t, supp_f in supp_fact_dict.items():
            class_train_data = train_data.loc[train_data[TARGET_LABEL] == t]
            supp_f_floor = int(supp_f)
            supp_f_deci = supp_f - supp_f_floor

            for _ in range(supp_f_floor):
                supp_train_data = pd.concat(
                    [supp_train_data, class_train_data], axis=0)

            num_last_samples = int(supp_f_deci * len(class_train_data.index))
            supp_train_data = pd.concat(
                [supp_train_data,
                 class_train_data.sample(num_last_samples)],
                axis=0)

        res = pd.concat([supp_train_data, train_data], axis=0)
        res = res.sample(frac=1)
        res_sampling_dict = {
            t: len(res.loc[res[TARGET_LABEL] == t].index)
            for t in TARGET_VALUES
        }
        print("res_sampling_dict: ", res_sampling_dict)

        return res

    def _display_cross_tab(self, test_target):

        cross_tab = pd.crosstab(test_target[TARGET_LABEL],
                                test_target['prediction'],
                                rownames=['Actual Target'],
                                colnames=['Predicted Target'])
        cross_tab_perc = cross_tab.apply(
            lambda r: (100 * round(r / r.sum(), 2)).astype(int), axis=1)
        cross_tab['Total'] = cross_tab.sum(axis=1)
        cross_tab = cross_tab.append(cross_tab.sum(), ignore_index=True)

        print("cross_tab: ", cross_tab)
        cross_tab_perc['Total'] = cross_tab['Total']
        cross_tab_perc = cross_tab_perc.append(cross_tab.iloc[-1],
                                               ignore_index=True)
        print("Cross Tab %: ", cross_tab_perc)

    def __init__(self, dir_path):

        self.dir_path = dir_path

        self.model = None
        self.model_params = None

    def start_model_type(self,
                         model_type,
                         model_prefix=None,
                         write_metrics=False):
        self.model_type = model_type
        self.model_prefix = model_prefix

        self.write_metrics = write_metrics

        if self.write_metrics:
            self._init_metrics()

    def generate_model(self, model_params, debug=False):

        if self.model_type is None:
            print("Error: Model type not provided for generation.")
            return None

        if self.model_type == ModelType.K_NN and self.model_prefix is None:
            print("Error: Model is of type K_NN but no prefix provided.")
            return None

        self.model_params = model_params
        self.model = generate_model(self.dir_path,
                                    self.model_type,
                                    self.model_params,
                                    model_debug=debug)

    def format_train_data(self, train_data):
        balanced_train_data = self._oversampling(train_data)

        train_input, train_target = self._format_input_target(
            balanced_train_data)

        return train_input, train_target

    def build_model(self, train_input, train_target):

        self.model.build_model(train_input, train_target)

        model_dict = dict()
        model_dict['type'] = self.model_type.name
        model_dict['prefix'] = self.model_prefix
        model_dict['params'] = self.model_params

        return self.model

    def evaluate_model(self, data_df, cl_dirpath):
        print("data_df: ", data_df)

        data_input, data_target = self._format_input_target(data_df)

        data_target['prediction'] = self.model.predict(data_input)
        test_proba = self.model.predict_proba(data_input)

        self._display_cross_tab(data_target)

        columns_labels = [
            self.model_type.name + '_' + proba_label
            for proba_label in COL_PROBA_NAMES
        ]
        test_proba_df = pd.DataFrame(test_proba,
                                     data_input.index,
                                     columns=columns_labels)

        print("test_proba_df: ", test_proba_df)

        data_target['proba'] = test_proba.tolist()
        log_loss = sm.log_loss(data_target[TARGET_LABEL], test_proba.tolist())
        accuracy_score = sm.accuracy_score(data_target[TARGET_LABEL],
                                           data_target['prediction'])

        eval_score_dict = dict()

        eval_score_dict['log_loss'] = log_loss
        eval_score_dict['accuracy_score'] = accuracy_score

        eval_rank = rank_proba(test_proba_df, self.model_type.name)
        eval_rank['era'] = data_df['era']

        model_eval_data = pd.concat([test_proba_df, eval_rank], axis=1)
        model_eval_fn = self.model_type.name + '_valid_data.csv'
        model_eval_fp = cl_dirpath + '/' + model_eval_fn
        with open(model_eval_fp, 'w') as fp:
            eval_score_dict['valid_data_fp'] = model_eval_fp
            model_eval_data.to_csv(fp)

        eval_score_dict['valid_score'] = valid_score(eval_rank,
                                                     self.model_type.name,
                                                     data_target[TARGET_LABEL])

        print("eval: ", eval_score_dict)

        if self.write_metrics:
            self._append_new_metrics(self.model, log_loss, accuracy_score)

        return eval_score_dict

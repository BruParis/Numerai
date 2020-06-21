import os
import sys
import numpy as np
import pandas as pd
import itertools

from pool_map import pool_map

from model_abstract import Model, ModelType
import model_itf


class ModelGenerator():

    def _init_metrics(self):
        metrics_filepath, metrics_header = model_itf.init_metrics(
            self.dir_path, self.model_type, self.model_prefix)

        with open(metrics_filepath, 'w') as f:
            metrics_header.to_csv(f, header=True, index=False)
            self.metrics_filepath = metrics_filepath

    def _append_new_metrics(self, model, log_loss, accuracy_score):

        if self.model_type is None:
            print("Error: File path absent to append new metrics.")
            return None

        metrics_df = model_itf.produce_metrics(
            model, log_loss, accuracy_score, self.model_prefix)

        with open(self.metrics_filepath, 'a') as f:
            metrics_df.to_csv(f, header=False, index=False)

    def __init__(self, dir_path):

        self.dir_path = dir_path

        self.model = None
        self.model_params = None

    def start_model_type(self, model_type, model_prefix=None, write_metrics=False):
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

        self.model = model_itf.generate_model(self.dir_path, self.model_type,
                                              self.model_params, model_debug=debug)

    def build_evaluate_model(self, train_data, test_data, debug=False):

        self.model.build_model(train_data)

        model_dict = dict()
        model_dict['type'] = self.model_type.name
        model_dict['prefix'] = self.model_prefix

        model_dict['params'] = self.model_params

        print("test_data: ", test_data)

        log_loss, accuracy_score = self.model.evaluate_model(test_data)

        if debug:
            print("log_loss: ", log_loss, " - accuracy_score: ", accuracy_score)

        if self.write_metrics:
            self._append_new_metrics(self.model, log_loss, accuracy_score)

        model_dict['log_loss'] = log_loss
        model_dict['accuracy_score'] = accuracy_score

        return self.model, model_dict

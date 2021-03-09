import keras
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import math
import json
import functools

from .model_abstract import Model, ModelType
from ..reader import ReaderCSV
from ..common import *
from ..data_analysis import rank_proba

BATCH_SIZE_PREDICT = 1000
MEAN = 0.5


class NeuralNetwork(Model):
    def _normalize_numeric_data(self, data, mean, std):
        return (data - mean) / std

    def _generate_layers_array(self):

        # NORMALIZE DATA -> produce mean and std from training data if necessary (?)
        # MEAN = [0.50 for i in range(self.n_features)]
        # STD = [-0.33 for i in range(self.n_features)]
        # normalizer = functools.partial(
        #    self._normalize_numeric_data, mean=MEAN, std=STD)

        # numeric_columns = tf.feature_column.numeric_column(
        #    'input', normalizer_fn=normalizer, shape=[self.n_features])

        # input_layer = keras.layers.DenseFeatures(numeric_columns)

        layers_size = [
            int(self.n_features *
                math.pow(self.model_params['size_factor'], i))
            for i in range(1, self.model_params['num_layers'] + 1)
        ]

        # use gelu instead of relu?

        # dense_layers_tuple = [[
        #     keras.layers.Dense(size, activation='relu'),
        #     keras.layers.Dropout(.2)
        # ] for size in layers_size]
        dense_layers_tuple = [[
            keras.layers.Dense(size, activation='sigmoid'),
            keras.layers.Dropout(.2)
        ] for size in layers_size]
        # dense_layers_tuple = [[
        #     keras.layers.Dense(size, activation='softmax'),
        #     keras.layers.Dropout(.2)
        # ] for size in layers_size]
        dense_layers = [
            layer for layer_tuple in dense_layers_tuple
            for layer in layer_tuple
        ]

        output_layers = [
            keras.layers.Dense(len(TARGET_VALUES),
                               activation='softmax',
                               name='output')
        ]

        full_layers = dense_layers + output_layers

        return full_layers

    def __init__(
        self,
        dirname,
        model_params=None,
        debug=False,
        filename=None,
    ):
        Model.__init__(self, ModelType.NeuralNetwork, dirname, model_params,
                       debug, filename)

    def _generate_target_labels(self, target_df):
        for v in TARGET_VALUES:
            target_df[v] = target_df.apply(lambda x: 1.0 if x[
                ('target')] == v * TARGET_FACT_NUMERIC else 0.0,
                                           axis=1)
        print("target_df: ", target_df)

        train_t_labels = target_df.loc[:, TARGET_VALUES]

        return train_t_labels

    def build_model(self, train_input, train_target, random_search=False):
        if self.debug:
            print("model params: ", self.model_params)

        if ERA_LABEL in train_input.columns:
            train_input = train_input.drop([ERA_LABEL], axis=1)

        if ERA_LABEL in train_target.columns:
            train_target = train_target.drop([ERA_LABEL], axis=1)

        train_input = train_input - MEAN

        print("Neural Net KERAS - build model")
        print("train_target: ", train_target)
        train_t_labels = self._generate_target_labels(train_target)

        self.n_features = len(train_input.columns)

        # model NeuralNet
        # need implement model structure with model_params
        # BUILD MODEL - TRAIN, EVALUATE
        full_layers = self._generate_layers_array()
        self.model = keras.Sequential(full_layers)

        # loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        self.model.compile(
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['accuracy'])

        # cp_callback = tf.keras.callbacks.ModelCheckpoint(self.filepath,
        #                                                  verbose=1,
        #                                                  save_weights_only=True)

        # CHOICE epochs 20 ? 30 ?
        # self.model.fit(packed_train_ds, epochs=self.model_params['num_epoch'])
        print("train_batch_size: ", self.model_params['train_batch_size'])
        self.model.fit(x=train_input,
                       batch_size=self.model_params['train_batch_size'],
                       y=train_t_labels,
                       epochs=self.model_params['num_epoch'])

    def predict(self, input_ds):
        predict_proba = self.predict_proba(input_ds)
        prediction = [np.argmax(proba) for proba in predict_proba]
        return prediction

    def predict_proba(self, data_input):
        data_input = data_input - MEAN

        prediction_proba = self.model.predict(data_input)

        return prediction_proba

    def save_model(self):
        self.model.save_weights(self.filepath)

        with open(self.config_filepath, 'w') as fp:
            json.dump(self.model_config(), fp, indent=4)

        return self.filepath, self.config_filepath

    def load_model(self):
        self._load_model_config()

        full_layers = self._generate_layers_array()
        self.model = keras.Sequential(full_layers)
        self.model.load_weights(self.filepath).expect_partial()

        # Or save wiuth saver : tf.train.Saver(tf.model_variables())

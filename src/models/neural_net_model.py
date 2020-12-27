from .model_abstract import Model, ModelType
from reader import ReaderCSV
from common import *
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import tensorflow as tf
import math
import json
import functools


BATCH_SIZE_TRAIN = 1000
BATCH_SIZE_PREDICT = 1000


class PackNumericFeaturesLabels(object):
    def __init__(self, names, label_values):
        self.names = names
        self.label_values = label_values

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32)
                            for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        input_features = {'input': numeric_features}

        # numeric_features = [tf.cast(feat, tf.float32)
        #                     for feat in features]
        # numeric_features = tf.stack(numeric_features, axis=-1)

        labels_multiclass = [tf.map_fn(lambda x: tf.cast(
            tf.math.equal(x[0], v), tf.float32), labels) for v in range(len(self.label_values))]
        labels_multiclass = tf.stack(labels_multiclass, axis=-1)

        # labels = tf.map_fn(lambda x: tf.cast(
        #     self._label_array(x), tf.float32), labels)

        return input_features, labels_multiclass


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    # @tf.autograph.experimental.do_not_convert
    def __call__(self, features):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32)
                            for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        input_features = {'input': numeric_features}

        return input_features


class NeuralNetwork(Model):

    def _show_batch(self, dataset):
        print("_show_batch - dataset: ", dataset)
        for batch, label, in dataset.take(1):
            print("batch: ", batch)
            print("label: ", label)

    def _pack_input_target_ds(self, input_df, target_df, batch_size=None):

        # if shuffle
        # dataset = dataset.shuffle(buffer_size=len(input_df))

        input_labels = input_df.columns.values

        dataset = tf.data.Dataset.from_tensor_slices(
            (dict(input_df), target_df))

        if batch_size is not None:
            dataset = dataset.batch(batch_size)

        packed_ds = dataset.map(
            PackNumericFeaturesLabels(input_labels, TARGET_VALUES))

        # self._show_batch(packed_ds)

        return packed_ds

    def _pack_input_ds(self, input_df, batch_size=None):

        input_labels = input_df.columns.values

        dataset = tf.data.Dataset.from_tensor_slices(dict(input_df))

        if batch_size is not None:
            dataset = dataset.batch(batch_size)

        packed_ds = dataset.map(
            PackNumericFeatures(input_labels))

        # self._show_batch(packed_ds)

        return packed_ds

    def _normalize_numeric_data(self, data, mean, std):
        return (data-mean)/std

    def _generate_layers_array(self):

        # NORMALIZE DATA -> produce mean and std from training data if necessary (?)
        MEAN = [0.50 for i in range(self.n_features)]
        STD = [-0.33 for i in range(self.n_features)]
        normalizer = functools.partial(
            self._normalize_numeric_data, mean=MEAN, std=STD)

        numeric_columns = tf.feature_column.numeric_column(
            'input', normalizer_fn=normalizer, shape=[self.n_features])
        # numeric_columns = tf.feature_column.numeric_column(
        #     'input', shape=[self.n_features])

        input_layer = tf.keras.layers.DenseFeatures(numeric_columns)

        layers_size = [int(self.n_features * math.pow(self.model_params['size_factor'], i))
                       for i in range(1, self.model_params['num_layers']+1)]

        # use gelu instead of relu?
        dense_layers_tuple = [[tf.keras.layers.Dense(size, activation='relu'),
                               tf.keras.layers.Dropout(.2)] for size in layers_size]
        dense_layers = [
            layer for layer_tuple in dense_layers_tuple for layer in layer_tuple]

        full_layers = [input_layer] + dense_layers + [tf.keras.layers.Dense(
            len(TARGET_VALUES), activation='softmax', name='output')]

        return full_layers

    def __init__(self, dirname, model_params=None, debug=False, filename=None,):
        Model.__init__(self, ModelType.NeuralNetwork,
                       dirname, model_params, debug, filename)

    def oversampling(self, train_data):
        # TODO : balance classes
        sampling_dict = {
            t: len(train_data.loc[train_data[TARGET_LABEL] == t].index) for t in TARGET_VALUES}
        print("sampling_dict: ", sampling_dict)
        max_class = max(sampling_dict, key=sampling_dict.get)
        print("max_class: ", max_class)
        supp_fact_dict = {k: (sampling_dict[max_class] - sampling_dict[k]) /
                          sampling_dict[k] for k, v in sampling_dict.items()}
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
                [supp_train_data, class_train_data.sample(num_last_samples)], axis=0)

        res = pd.concat([supp_train_data, train_data], axis=0)
        res = res.sample(frac=1)
        res_sampling_dict = {
            t: len(res.loc[res[TARGET_LABEL] == t].index) for t in TARGET_VALUES}
        print("res_sampling_dict: ", res_sampling_dict)

        return res

    def build_model(self, train_data):
        if train_data is None:
            print("No train data provided!")
            return

        if self.debug:
            print("model params: ", self.model_params)

        balanced_train_data = self.oversampling(train_data)

        train_input, train_target = self._format_input_target(
            balanced_train_data)
        packed_train_ds = self._pack_input_target_ds(
            train_input, train_target, self.model_params['train_batch_size'])

        self.n_features = len(train_input.columns)

        # model NeuralNet
        # need implement model structure with model_params
        # BUILD MODEL - TRAIN, EVALUATE
        full_layers = self._generate_layers_array()
        self.model = tf.keras.Sequential(full_layers)

        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['accuracy']
        )

        # cp_callback = tf.keras.callbacks.ModelCheckpoint(self.filepath,
        #                                                  verbose=1,
        #                                                  save_weights_only=True)

        # , callbacks=[cp_callback])
        # CHOICE epochs 20 ? 30 ?
        self.model.fit(packed_train_ds,
                       epochs=self.model_params['num_epoch'])

    def evaluate_model(self, test_data):
        test_input, test_target = self._format_input_target(test_data)
        packed_test_input_target_ds = self._pack_input_target_ds(
            test_input, test_target, BATCH_SIZE_PREDICT)

        test_target['prediction'] = self.predict(test_input)

        test_loss, test_accuracy = self.model.evaluate(
            packed_test_input_target_ds)
        if self.debug:
            print('\n\nTest Loss {}, Test Accuracy {}'.format(
                test_loss, test_accuracy))
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

        return test_loss, test_accuracy

    def predict(self, input_ds):
        predict_proba = self.predict_proba(input_ds)
        prediction = [np.argmax(proba) for proba in predict_proba]
        return prediction

    def predict_proba(self, data_input):
        packed_input_ds = self._pack_input_ds(data_input, BATCH_SIZE_PREDICT)
        prediction_proba = self.model.predict(
            packed_input_ds, batch_size=BATCH_SIZE_PREDICT)

        # check if proba sum up to 1 -> softmax funct. as output
        # proba_sum = prediction_proba.sum(axis=1)
        # print("proba_sum: ", proba_sum)

        return prediction_proba

    def save_model(self):
        self.model.save_weights(self.filepath)

        with open(self.config_filepath, 'w') as fp:
            json.dump(self.model_config(), fp, indent=4)

        return self.filepath, self.config_filepath

    def load_model(self):
        self._load_model_config()

        full_layers = self._generate_layers_array()
        self.model = tf.keras.Sequential(full_layers)
        self.model.load_weights(self.filepath)

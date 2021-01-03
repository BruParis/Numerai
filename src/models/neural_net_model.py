from .model_abstract import Model, ModelType
from reader import ReaderCSV
from common import *
import keras
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import math
import json
import functools

BATCH_SIZE_PREDICT = 1000


class NeuralNetwork(Model):

    def _normalize_numeric_data(self, data, mean, std):
        return (data-mean)/std

    def _generate_layers_array(self):

        # NORMALIZE DATA -> produce mean and std from training data if necessary (?)
        # MEAN = [0.50 for i in range(self.n_features)]
        # STD = [-0.33 for i in range(self.n_features)]
        # normalizer = functools.partial(
        #    self._normalize_numeric_data, mean=MEAN, std=STD)

        # numeric_columns = tf.feature_column.numeric_column(
        #    'input', normalizer_fn=normalizer, shape=[self.n_features])

        # input_layer = keras.layers.DenseFeatures(numeric_columns)

        layers_size = [int(self.n_features * math.pow(self.model_params['size_factor'], i))
                       for i in range(1, self.model_params['num_layers']+1)]

        # use gelu instead of relu?
        dense_layers_tuple = [[keras.layers.Dense(size, activation='relu'),
                               keras.layers.Dropout(.2)] for size in layers_size]
        dense_layers = [
            layer for layer_tuple in dense_layers_tuple for layer in layer_tuple]

        output_layers = [keras.layers.Dense(len(TARGET_VALUES), activation='softmax', name='output')]

        # full_layers = [input_layer] + dense_layers + [keras.layers.Dense(
        #     len(TARGET_VALUES), activation='softmax', name='output')]
        full_layers = dense_layers + output_layers

        return full_layers

    def __init__(self, dirname, model_params=None, debug=False, filename=None,):
        Model.__init__(self, ModelType.NeuralNetwork,
                       dirname, model_params, debug, filename)

    def _oversampling(self, train_data):
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

    def _generate_target_labels(self, target_df):
        for v in TARGET_VALUES:
            target_df[v] = target_df.apply(
                lambda x: 1.0 if x[('target')] == v * TARGET_FACT_NUMERIC
                else 0.0, axis=1)
        print("target_df: ", target_df)

        train_t_labels = target_df.loc[:, TARGET_VALUES]

        return train_t_labels

    def build_model(self, train_data):
        if train_data is None:
            print("No train data provided!")
            return

        if self.debug:
            print("model params: ", self.model_params)

        print("Neural Net KERAS - build model")

        balanced_train_data = self._oversampling(train_data)

        train_input, train_target = self._format_input_target(
            balanced_train_data)

        print("train_target: ", train_target)
        train_t_labels = self._generate_target_labels(train_target)

        # packed_train_ds = self._pack_input_target_ds(
        #     train_input, train_target, self.model_params['train_batch_size'])

        self.n_features = len(train_input.columns)

        # model NeuralNet
        # need implement model structure with model_params
        # BUILD MODEL - TRAIN, EVALUATE
        full_layers = self._generate_layers_array()
        self.model = keras.Sequential(full_layers)

        self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['accuracy']
        )

        # cp_callback = tf.keras.callbacks.ModelCheckpoint(self.filepath,
        #                                                  verbose=1,
        #                                                  save_weights_only=True)

        # , callbacks=[cp_callback])
        # CHOICE epochs 20 ? 30 ?
        # self.model.fit(packed_train_ds, epochs=self.model_params['num_epoch'])
        print("train_batch_size: ", self.model_params['train_batch_size'])
        self.model.fit(x=train_input, batch_size=self.model_params['train_batch_size'],
                        y=train_t_labels, epochs=self.model_params['num_epoch'])

    def evaluate_model(self, test_data):
        test_input, test_target = self._format_input_target(test_data)
        test_t_labels = self._generate_target_labels(test_target)

        test_target['prediction'] = self.predict(test_input)

        test_loss, test_accuracy = self.model.evaluate(
            x=test_input, y=test_t_labels)
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
        #packed_input_ds = self._pack_input_ds(data_input, BATCH_SIZE_PREDICT)
        # prediction_proba = self.model.predict(
        #     packed_input_ds, batch_size=BATCH_SIZE_PREDICT)
        print("data_input: ", data_input)
        prediction_proba = self.model.predict(data_input)

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
        self.model = keras.Sequential(full_layers)
        self.model.load_weights(self.filepath)

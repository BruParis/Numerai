import keras
import matplotlib.pyplot as plt
import matplotlib.table as table
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import os
import math
import json
import errno
import functools

from .model_abstract import Model, ModelType
from ..reader import ReaderCSV
from ..common import *
from ..data_analysis import rank_proba

ERA_NUM_BATCH = 100
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
                math.pow(self.model_params["size_factor"], i))
            for i in range(1, self.model_params["num_layers"] + 1)
        ]

        dense_layers_t = [(keras.layers.Dense(
            size, activation=self.model_params["func_activation"]),
                           keras.layers.Dropout(0.2)) for size in layers_size]

        dense_layers = [
            layer for layer_tuple in dense_layers_t for layer in layer_tuple
        ]

        output_layers = [
            keras.layers.Dense(len(TARGET_VALUES),
                               activation="softmax",
                               name="output")
        ]

        full_layers = dense_layers + output_layers

        return full_layers

    def _plot_history(self, hist_fp):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15))
        ax1.plot(self.history.history["accuracy"])
        ax1.plot(self.history.history["val_accuracy"])
        ax1.title.set_text("model accuracy")
        ax1.set_ylabel("accuracy")
        ax1.set_xlabel("epoch")
        ax1.legend(["train", "test"], loc="upper left")

        ax2.plot(self.history.history["loss"])
        ax2.plot(self.history.history["val_loss"])
        ax2.title.set_text("model loss")
        ax2.set_ylabel("loss")
        ax2.set_xlabel("epoch")
        ax2.legend(["train", "test"], loc="upper left")

        celltext = [[str(p)] for p in self.model_params.values()]
        table.table(
            ax2,
            cellText=celltext,
            rowLabels=[str(k) for k in self.model_params.keys()],
        )

        fig.savefig(hist_fp)

        self.model_params

    def __init__(
        self,
        dirname,
        model_params=None,
        debug=False,
        filename=None,
    ):
        Model.__init__(self, ModelType.NeuralNetwork, dirname, model_params,
                       debug, filename)
        self.history = None

    def _generate_target_labels(self, target_df):
        # target_df reference data that can be used by other models
        target_nn_df = target_df.copy()
        for v in TARGET_VALUES:
            target_nn_df[v] = target_nn_df.apply(
                lambda x: 1.0
                if x[("target")] == v * TARGET_FACT_NUMERIC else 0.0,
                axis=1,
            )
        print("target_nn_df: ", target_nn_df)

        train_t_labels = target_nn_df.loc[:, TARGET_VALUES]

        return train_t_labels

    def build_model(self, train_input, train_target, random_search=False):
        if self.debug:
            print("model params: ", self.model_params)

        if random_search:
            print("random search is not available for NeuralNetwork")
            return

        print("Neural Net KERAS - build model")
        print("train_target: ", train_target)

        self.n_features = len(train_input.columns)

        full_layers = self._generate_layers_array()
        self.model = keras.Sequential(full_layers)

        # loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=["accuracy"],
        )

        # cp_callback = tf.keras.callbacks.ModelCheckpoint(self.filepath,
        #                                                  verbose=1,
        #                                                  save_weights_only=True)

        # CHOICE epochs 20 ? 30 ?

        num_eras = self.model_params['num_eras']
        tr_batch_size = int(
            len(train_input.index) / (ERA_NUM_BATCH * num_eras))

        self.model_params["train_batch_size"] = tr_batch_size
        print("model_params: ", self.model_params)
        if ERA_LABEL in train_input.columns:
            train_input = train_input.drop([ERA_LABEL], axis=1)

        if ERA_LABEL in train_target.columns:
            train_target = train_target.drop([ERA_LABEL], axis=1)

        train_input = train_input - MEAN

        train_t_labels = self._generate_target_labels(train_target)

        stop_callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss',
                                          patience=15,
                                          mode='auto'),
            keras.callbacks.ModelCheckpoint(self.filepath,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_weights_only=True)
        ]

        self.history = self.model.fit(
            x=train_input,
            y=train_t_labels,
            callbacks=stop_callbacks,
            validation_split=0.20,
            batch_size=self.model_params["train_batch_size"],
            epochs=self.model_params["num_epoch"],
        )

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

        with open(self.config_filepath, "w") as fp:
            json.dump(self.model_config(), fp, indent=4)

        return self.filepath, self.config_filepath

    def load_model(self):
        self._load_model_config()

        full_layers = self._generate_layers_array()
        self.model = keras.Sequential(full_layers)
        self.model.load_weights(self.filepath).expect_partial()

        # Or save wiuth saver : tf.train.Saver(tf.model_variables())

    def save_hist_plot(self, suffix=None):

        if suffix is None:
            hist_fp = self.dirname + "/" + "hist_train.png"
            self._plot_history(hist_fp)
            return
        else:
            hist_dirpath = self.dirname + "/NN_historics"
            try:
                os.makedirs(hist_dirpath)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    print(f"error creating dir {hist_dirpath}")
                    return

            hist_fp = hist_dirpath + "/" + "hist_" + str(suffix) + ".png"
            self._plot_history(hist_fp)
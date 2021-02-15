from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sm
import pandas as pd
import pickle
import json

#from reader import ReaderCSV
from .model_abstract import Model, ModelType
from ..common import *


class RFModel(Model):
    def __init__(self, dirname, model_params=None, debug=False, filename=None):
        Model.__init__(self, ModelType.RandomForest, dirname, model_params,
                       debug, filename)

    def build_model(self, train_input, train_target):
        if self.debug:
            print("model params: ", self.model_params)

        # check model_params structure coherent ith RandomForest (?)

        self.model = RandomForestClassifier(
            n_estimators=self.model_params['n_estimators'],
            max_depth=self.model_params['max_depth'],
            min_samples_split=10,
            min_samples_leaf=4,
            warm_start=True,
            random_state=0)

        self.n_features = len(train_input.columns)

        self.model.fit(train_input, train_target.values.ravel())

    def predict(self, data_input):
        prediction = self.model.predict(data_input)
        return prediction

    def predict_proba(self, data_input):
        prediction = self.model.predict_proba(data_input)
        return prediction


# clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
#                                 n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
# Fit the random search model
# clf_random.fit(train_input, train_target)
# print("clf_random basat params: ", clf_random.best_params_)

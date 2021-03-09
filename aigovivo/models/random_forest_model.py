import numpy as np
from sklearn.model_selection import KFold
import random
import itertools

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sm
import pandas as pd
import pickle
import json

from ..common import *

from .model_abstract import Model, ModelType
from .k_folds_by_era import cv_k_folds_by_era

NUM_FOLDS = 5


class RFModel(Model):
    def _create_random_grid(self):

        # Number of trees in random forest
        #n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        n_estimators = [int(x) for x in np.linspace(start=60, stop=120, num=5)]
        # Number of features to consider at every split
        #max_features = ['auto', 'sqrt']
        max_features = ['sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(70, 130, num=6)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        # min_samples_split = [2, 5, 10]
        min_samples_split = [5]
        # Minimum number of samples required at each leaf node
        # min_samples_leaf = [1, 2, 4]
        min_samples_leaf = [1]
        # Method of selecting samples for training each tree
        # bootstrap = [True, False]
        bootstrap = [False]
        # Create the random grid
        random_grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }
        print(random_grid)

        return random_grid

    def __init__(self, dirname, model_params=None, debug=False, filename=None):
        Model.__init__(self, ModelType.RandomForest, dirname, model_params,
                       debug, filename)

    def build_model(self, train_input, train_target, random_search=False):
        if self.debug:
            print("model params: ", self.model_params)

        # check model_params structure coherent ith RandomForest (?)

        self.model = RandomForestClassifier(
            n_estimators=self.model_params['n_estimators'],
            max_depth=self.model_params['max_depth'],
            min_samples_split=5,
            min_samples_leaf=1,
            warm_start=True,
            random_state=0)

        self.n_features = len(train_input.columns)

        if random_search:
            print("Fit the random search model")
            train_input = train_input.reset_index(drop=True)
            train_target = train_target.reset_index(drop=True)

            # cross validation split for random search
            cv_strat = cv_k_folds_by_era(train_input, num_folds=NUM_FOLDS)

            random_grid = self._create_random_grid()
            clf_random = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=random_grid,
                n_iter=15,
                cv=cv_strat,
                # cv=5,
                verbose=1,
                random_state=42,
                n_jobs=-1)

            if ERA_LABEL in train_input.columns:
                train_input = train_input.drop([ERA_LABEL], axis=1)

            clf_random.fit(train_input, train_target.values.ravel())

            print("clf_random best params: ", clf_random.best_params_)
            self.model_params = clf_random.best_params_
            self.model = clf_random.best_estimator_
        else:
            if ERA_LABEL in train_input.columns:
                train_input = train_input.drop([ERA_LABEL], axis=1)

            self.model.fit(train_input, train_target.values.ravel())

    def predict(self, data_input):
        prediction = self.model.predict(data_input)
        return prediction

    def predict_proba(self, data_input):
        prediction = self.model.predict_proba(data_input)
        return prediction
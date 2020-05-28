from reader_csv import ReaderCSV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import pickle
import json

from model_abstract import Model, ModelType


class RFModel(Model):

    @staticmethod
    def create_random_grid():

        # Number of trees in random forest
        # n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        n_estimators = [int(x)
                        for x in np.linspace(start=200, stop=300, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        # min_samples_split = [2, 5, 10]
        min_samples_split = [10]
        # Minimum number of samples required at each leaf node
        # min_samples_leaf = [1, 2, 4]
        min_samples_leaf = [4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print(random_grid)

        return random_grid

    def __init__(self, dirname, train_data=None, test_data=None, model_params=None, debug=False, filename=None):
        Model.__init__(self, ModelType.RandomForest,
                       dirname, train_data, test_data, model_params, debug, filename)

    def build_model(self):
        if self.train_data is None:
            print("No train data provided!")
            return

        if self.debug:
            print("model params: ", self.model_params)

        # check model_params structure coherent ith RandomForest (?)

        self.model = RandomForestClassifier(n_estimators=self.model_params['n_estimators'],
                                            max_depth=self.model_params['max_depth'],
                                            min_samples_split=10,
                                            min_samples_leaf=4,
                                            random_state=0)

        train_input, train_target = self._format_input_target(self.train_data)

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

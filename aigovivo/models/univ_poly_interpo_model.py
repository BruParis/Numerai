from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from .model_abstract import Model, ModelType
from ..common import *


#Univariate polynomial interpolation
class UnivPolyInterpo(Model):
    def __init__(self, dirname, model_params=None, debug=False, filename=None):
        Model.__init__(self, ModelType.UnivPolyInterpo, dirname, model_params,
                       debug, filename)

    def build_model(self, train_input, train_target, random_search=False):
        if self.debug:
            print("model params: ", self.model_params)

        self.n_features = 1

        self.model = make_pipeline(
            PolynomialFeatures(self.model_params['degree']), Ridge())

        train_input_array = train_input.to_numpy()
        print("train_input_array: ", train_input_array)
        # reshape input
        train_input_array = train_input_array.reshape(-1, 1)
        print("reshape train_input_array: ", train_input_array)
        self.model.fit(train_input_array, train_target.values.ravel())

    def predict(self, data_input):
        prediction = self.model.predict(data_input)
        return prediction

    def predict_proba(self, data_input):
        print("len data_input: ", len(data_input))
        print("data_input: ", data_input)
        prediction = self.model.predict(data_input)
        print("len data_input: ", len(prediction))
        print("prediction: ", prediction)

        return prediction

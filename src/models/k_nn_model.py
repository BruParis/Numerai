from reader import ReaderCSV
from sklearn.neighbors import KNeighborsClassifier

from .model_abstract import Model, ModelType


class K_NNModel(Model):
    def __init__(self, dirname, model_params=None, debug=False, filename=None):
        Model.__init__(self, ModelType.K_NN, dirname, model_params, debug,
                       filename)

    def build_model(self, train_input, train_target):

        if self.debug:
            print("model params: ", self.model_params)

        # model_param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
        # model_param['eval_metric'] = 'auc'

        # evallist = [(self.test_data, 'eval'), (self.train_data, 'train')]
        # num_round = model_params['num_round'] -> best tree limit
        # XGBRFClassifier (?)
        self.model = KNeighborsClassifier(
            n_neighbors=self.model_params['n_neighbors'],
            leaf_size=self.model_params['leaf_size'],
            p=self.model_params['minkowski_dist'])

        self.n_features = len(train_input.columns)

        self.model.fit(train_input, train_target.values.ravel())

    def predict(self, data_input):
        prediction = self.model.predict(data_input)
        return prediction

    def predict_proba(self, data_input):
        prediction = self.model.predict_proba(data_input)
        return prediction

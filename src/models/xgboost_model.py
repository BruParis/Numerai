from reader import ReaderCSV
import sklearn.metrics as sm
import xgboost as xgb

from common import *
from .model_abstract import Model, ModelType


class XGBModel(Model):
    def __init__(self, dirname, model_params=None, debug=False, filename=None):

        Model.__init__(self, ModelType.XGBoost, dirname, model_params, debug,
                       filename)

    def build_model(self, train_input, train_target):

        if self.debug:
            print("model params: ", self.model_params)

        # model_param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
        # model_param['eval_metric'] = 'auc'

        # evallist = [(self.test_data, 'eval'), (self.train_data, 'train')]
        # num_round = model_params['num_round'] -> best tree limit
        # XGBRFClassifier (?)
        # objective='binary:logisitc',
        # objective='multi:softprob',
        self.model = xgb.XGBClassifier(
            objective='binary:logisitc',
            eval_metric='mlogloss',
            num_class=len(TARGET_CLASSES),
            n_estimators=self.model_params['n_estimators'],
            max_depth=self.model_params['max_depth'],
            learning_rate=self.model_params['learning_rate'],
            booster='gbtree',
            scale_pos_weight=SCALE_POS_WEIGHT,
            n_jobs=-1)

        self.n_features = len(train_input.columns)

        train_target_r = train_target.values.ravel()

        print("START TRAINING")
        self.model.fit(train_input, train_target_r)

        print("DONE")

    def predict(self, data_input):
        prediction = self.model.predict(data_input)
        return prediction

    def predict_proba(self, data_input):
        prediction = self.model.predict_proba(data_input)
        return prediction

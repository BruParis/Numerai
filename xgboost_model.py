from reader_csv import ReaderCSV
import xgboost as xgb

from model_abstract import Model, ModelType, TARGET_VALUES, ERA_LABEL


class XGBModel(Model):

    def __init__(self, dirname, train_data=None, test_data=None,
                 model_params=None, debug=False, filename=None):

        if train_data is not None:
            train_data = train_data.drop([ERA_LABEL], axis=1)
            self.train_data = xgb.DMatrix(train_data, label=TARGET_VALUES)

        if test_data is not None:
            test_data = test_data.drop([ERA_LABEL], axis=1)
            self.test_data = xgb.DMatrix(test_data, label=TARGET_VALUES)

        Model.__init__(self, ModelType.XGBoost,
                       dirname, train_data, test_data, model_params, debug, filename)

    def build_model(self):
        if self.train_data is None:
            print("No train data provided!")
            return

        if self.debug:
            print("produce metrics for model params: ", self.model_params)

        # model_param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
        # model_param['eval_metric'] = 'auc'

        # evallist = [(self.test_data, 'eval'), (self.train_data, 'train')]
        # num_round = model_params['num_round'] -> best tree limit
        # XGBRFClassifier (?)
        self.model = xgb.XGBClassifier(objective='binary:logisitc',
                                       n_estimators=self.model_params['n_estimators'],
                                       max_depth=self.model_params['max_depth'],
                                       learning_rate=self.model_params['learning_rate'],
                                       booster='gbtree',
                                       n_jobs=-1)
        # train(..., eval_metric=evals, early_stopping_rounds=10)
        train_input, train_target = self._format_input_target(self.train_data)

        self.n_features = len(train_input.columns)

        train_target_r = train_target.values.ravel()

        if self.debug:
            test_input, test_target = self._format_input_target(
                self.test_data)
            test_target_r = test_target.values.ravel()
            eval_set = [(train_input, train_target_r),
                        (test_input, test_target_r)]

            self.model.fit(train_input, train_target_r, eval_set=eval_set)
        else:
            self.model.fit(train_input, train_target_r)

    def predict(self, data_input):
        prediction = self.model.predict(data_input)
        return prediction

    def predict_proba(self, data_input):
        prediction = self.model.predict_proba(data_input)
        return prediction

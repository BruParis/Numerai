from reader_csv import ReaderCSV
import sklearn.metrics as sm
import xgboost as xgb

from model_abstract import Model, ModelType, TARGET_VALUES, ERA_LABEL, TARGET_LABEL


class XGBModel(Model):

    def _xgbformat_data(self, data):
        data = data.drop([ERA_LABEL], axis=1)
        res = xgb.DMatrix(data, label=TARGET_VALUES)

        return res

    def __init__(self, dirname, model_params=None, debug=False, filename=None):

        Model.__init__(self, ModelType.XGBoost, dirname,
                       model_params, debug, filename)

    def build_model(self, train_data_pd):
        if train_data_pd is None:
            print("No train data provided!")
            return

        if self.debug:
            print("model params: ", self.model_params)

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
        # train_data_format = self._xgbformat_data(train_data_pd)

        train_input, train_target = self._format_input_target(
            train_data_pd)

        self.n_features = len(train_input.columns)

        train_target_r = train_target.values.ravel()

        print("START TRAINING")
        self.model.fit(train_input, train_target_r)

        print("DONE")

    def evaluate_model(self, test_data_pd):
        if self.model is None:
            print("Model not yet built!")
            return

        if test_data_pd is None:
            print("No test data provided!")
            return

        test_input, test_target = self._format_input_target(test_data_pd)

        test_target['prediction'] = self.predict(test_input)

        if self.debug:
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

        return log_loss, accuracy_score

    def predict(self, data_input):
        prediction = self.model.predict(data_input)
        return prediction

    def predict_proba(self, data_input):
        prediction = self.model.predict_proba(data_input)
        return prediction

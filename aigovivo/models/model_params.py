import itertools
import numpy as np

from .model_abstract import Model, ModelType

# Random Forest
# FST ESTIMATION FOR PARAMTERS BOUNDS :
# n_est : < 268;340
# m_depth :  < 28;32
# min_splt : >= 10
# min_leaf : >= 4


def interpo_params():
    degree = [2]
    # L2 norm ? Manhattan ? max ?

    model_params_array = map(lambda x: {
        'degree': int(degree[0]),
    }, itertools.product(*[degree]))

    return model_params_array


def model_params(layer, eModel, metrics=False, model_prefix=None):

    if layer == 'fst':
        return make_fst_layer_model_params(eModel, metrics, model_prefix=None)
    elif layer == 'snd':
        return make_snd_layer_model_params(eModel, metrics, model_prefix=None)


def make_fst_layer_model_params(eModel, metrics, model_prefix=None):
    if eModel == ModelType.RandomForest:
        n_est = np.linspace(start=180, stop=250, num=20) if metrics else [120]
        max_d = np.linspace(10, 20, num=5) if metrics else [118]
        # min sample split
        model_params_array = map(
            lambda x: {
                'n_estimators': int(x[0]),
                'max_depth': int(x[1])
            }, itertools.product(*[n_est, max_d]))

        return model_params_array

    if eModel == ModelType.XGBoost:
        n_est = np.linspace(start=180, stop=250,
                            num=10) if metrics else [120]  #[120]
        max_d = np.linspace(10, 20, num=5) if metrics else [10]  # [118]
        # eta = learning_rate
        eta = np.logspace(start=(-1.0), stop=0.0, base=10.0,
                          num=5) if metrics else [1.0]
        # min sample split

        model_params_array = map(
            lambda x: {
                'n_estimators': int(x[0]),
                'max_depth': int(x[1]),
                'learning_rate': float(x[2])
            }, itertools.product(*[n_est, max_d, eta]))

        return model_params_array

    if eModel == ModelType.NeuralNetwork:
        num_layers = range(1, 3) if metrics else [1]
        layer_size_factor = np.linspace(start=0.33, stop=1,
                                        num=5) if metrics else [1]
        # train_batch_size = np.linspace(start=10, stop=100, num=4)  if metrics else [100]
        func_activation = ['sigmoid', 'relu', 'softmax'
                           ] if metrics else ['sigmoid']
        # use gelu isteand of relu ?
        num_epoch = [100]

        model_params_array = map(
            lambda x: {
                'num_layers': int(x[0]),
                'size_factor': float(x[1]),
                'func_activation': x[2],
                'num_epoch': int(x[3])
            },
            itertools.product(
                *[num_layers, layer_size_factor, func_activation, num_epoch]))

        return model_params_array

    if eModel == ModelType.K_NN:
        leaf_size = np.linspace(start=20, stop=50, num=1) if metrics else [20]
        minkowski_dist = [1]  # [1, 2, 3, 4]
        model_params_array = map(
            lambda x: {
                'n_neighbors': int(model_prefix),
                'leaf_size': int(x[0]),
                'minkowski_dist': int(x[1])
            }, itertools.product(*[leaf_size, minkowski_dist]))

        return model_params_array


def make_snd_layer_model_params(eModel, metrics, model_prefix=None):
    if eModel == ModelType.RandomForest:
        n_est = np.linspace(start=180, stop=250, num=20) if metrics else [120]
        max_d = np.linspace(10, 20, num=5) if metrics else [118]
        model_params_array = map(
            lambda x: {
                'n_estimators': int(x[0]),
                'max_depth': int(x[1])
            }, itertools.product(*[n_est, max_d]))

        return model_params_array

    if eModel == ModelType.XGBoost:
        n_est = np.linspace(start=180, stop=250, num=20) if metrics else [120]
        max_d = np.linspace(10, 20, num=5) if metrics else [118]
        # eta = learning_rate
        eta = [1.0]  # np.logspace(start=(-1.0), stop=0.0, base=10.0, num=5)

        model_params_array = map(
            lambda x: {
                'n_estimators': int(x[0]),
                'max_depth': int(x[1]),
                'learning_rate': float(x[2])
            }, itertools.product(*[n_est, max_d, eta]))

        return model_params_array

    if eModel == ModelType.NeuralNetwork:
        num_layers = np.linspace(start=1, stop=4) if metrics else [1]
        layer_size_factor = np.linspace(start=0.33, stop=1,
                                        num=10) if metrics else [0.66]
        # train_batch_size = [500]
        num_epoch = [25]
        model_params_array = map(
            lambda x: {
                'num_layers': int(x[0]),
                'size_factor': float(x[1]),
                'num_epoch': int(x[2])
            }, itertools.product(*[num_layers, layer_size_factor, num_epoch]))

        return model_params_array

    if eModel == ModelType.K_NN:
        leaf_size = [20]  # np.linspace(start=20, stop=50, num=1)
        minkowski_dist = [1]  # [1, 2, 3, 4]
        model_params_array = map(
            lambda x: {
                'n_neighbors': int(model_prefix),
                'leaf_size': int(x[0]),
                'minkowski_dist': int(x[1])
            }, itertools.product(*[leaf_size, minkowski_dist]))

        return model_params_array

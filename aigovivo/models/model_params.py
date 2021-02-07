import itertools

from .model_abstract import Model, ModelType

# Random Forest
# FST ESTIMATION FOR PARAMTERS BOUNDS :
# n_est : < 268;340
# m_depth :  < 28;32
# min_splt : >= 10
# min_leaf : >= 4


def model_params(layer, eModel, model_prefix=None):

    if layer == 'fst':
        return make_fst_layer_model_params(eModel, model_prefix=None)
    elif layer == 'snd':
        return make_snd_layer_model_params(eModel, model_prefix=None)


def make_fst_layer_model_params(eModel, model_prefix=None):
    if eModel == ModelType.RandomForest:
        n_est = [220]  # np.linspace(start=180, stop=250, num=20)
        max_d = [10]  # np.linspace(10, 20, num=5)
        model_params_array = map(
            lambda x: {
                'n_estimators': int(x[0]),
                'max_depth': int(x[1])
            }, itertools.product(*[n_est, max_d]))

        return model_params_array

    if eModel == ModelType.XGBoost:
        n_est = [220]  # np.linspace(start=180, stop=250, num=10)
        max_d = [10]  # np.linspace(10, 20, num=5)
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
        num_layers = [1]  # np.linspace(start=1, stop=4, num=1)
        layer_size_factor = [0.66]
        train_batch_size = [50]
        num_epoch = [30]
        model_params_array = map(
            lambda x: {
                'num_layers': int(x[0]),
                'size_factor': float(x[1]),
                'train_batch_size': int(x[2]),
                'num_epoch': int(x[3])
            },
            itertools.product(
                *[num_layers, layer_size_factor, train_batch_size, num_epoch]))

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


def make_snd_layer_model_params(eModel, model_prefix=None):
    if eModel == ModelType.RandomForest:
        n_est = [150]  # np.linspace(start=180, stop=250, num=7)
        max_d = [10]  # np.linspace(10, 20, num=5)
        model_params_array = map(
            lambda x: {
                'n_estimators': int(x[0]),
                'max_depth': int(x[1])
            }, itertools.product(*[n_est, max_d]))

        return model_params_array

    if eModel == ModelType.XGBoost:
        n_est = [150]  # np.linspace(start=180, stop=250, num=7)
        max_d = [5]  # np.linspace(5, 20, num=5)
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
        num_layers = [1]  # np.linspace(start=1, stop=4, num=1)
        layer_size_factor = [0.66]  # [0.33, 0.5, 0.66]
        train_batch_size = [500]
        num_epoch = [25]
        model_params_array = map(
            lambda x: {
                'num_layers': int(x[0]),
                'size_factor': float(x[1]),
                'train_batch_size': int(x[2]),
                'num_epoch': int(x[3])
            },
            itertools.product(
                *[num_layers, layer_size_factor, train_batch_size, num_epoch]))

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
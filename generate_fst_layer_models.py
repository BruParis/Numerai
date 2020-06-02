import os
import sys
import json
import numpy as np
import itertools
import json
import time

from pool_map import pool_map

from reader_csv import ReaderCSV
from model_abstract import Model, ModelType
from model_generator import ModelGenerator

# Random Forest
# FST ESTIMATION FOR PARAMTERS BOUNDS :
# n_est : < 268;340
# m_depth :  < 28;32
# min_splt : >= 10
# min_leaf : >= 4


def load_data(data_filename):
    file_reader = ReaderCSV(data_filename)
    data_df = file_reader.read_csv().set_index("id")

    return data_df


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


def make_model_params(eModel, model_prefix=None):
    if eModel == ModelType.RandomForest:
        n_est = [220]  # np.linspace(start=180, stop=250, num=20)
        max_d = [10]  # np.linspace(10, 20, num=5)
        model_params_array = map(lambda x: {'n_estimators': int(
            x[0]), 'max_depth': int(x[1])}, itertools.product(*[n_est, max_d]))

        return model_params_array

    if eModel == ModelType.XGBoost:
        n_est = [220]  # np.linspace(start=180, stop=250, num=10)
        max_d = [10]  # np.linspace(10, 20, num=5)
        # eta = learning_rate
        eta = [1.0]  # np.logspace(start=(-1.0), stop=0.0, base=10.0, num=5)

        model_params_array = map(lambda x: {'n_estimators': int(x[0]), 'max_depth': int(
            x[1]), 'learning_rate': float(x[2])}, itertools.product(*[n_est, max_d, eta]))

        return model_params_array

    if eModel == ModelType.NeuralNetwork:
        num_layers = [1]  # np.linspace(start=1, stop=4, num=1)
        layer_size_factor = [0.66]
        model_params_array = map(
            lambda x: {'num_layers': int(x[0]), 'size_factor': float(
                x[1])}, itertools.product(*[num_layers, layer_size_factor]))

        return model_params_array

    if eModel == ModelType.K_NN:
        leaf_size = [20]  # np.linspace(start=20, stop=50, num=1)
        minkowski_dist = [1]  # [1, 2, 3, 4]
        model_params_array = map(
            lambda x: {'n_neighbors': int(model_prefix), 'leaf_size': int(x[0]), 'minkowski_dist': int(
                x[1])}, itertools.product(*[leaf_size, minkowski_dist]))

        return model_params_array


def make_model_prefix(eModel):
    if eModel is not ModelType.K_NN:
        return [None]

    return [5, 10, 30]


def subset_model_build(dirname, sub_dirname, bSaveModel=False, bMetrics=False, model_debug=False):

    sub_dir_path = dirname + '/' + sub_dirname

    train_filepath = sub_dir_path + '/' + 'training_data.csv'
    test_filepath = sub_dir_path + '/' + 'test_data.csv'

    train_data = load_data(train_filepath)
    test_data = load_data(test_filepath)

    # bMultiProc = False
    # if bMultiProc:
    # Parallelism only shows a slight speed improvment
    # -> not enough memory for mutliple thread w/ full dataset (?)
    # model_metrics_arg = list(zip(itertools.repeat(sub_dir_path), itertools.repeat(
    #     metrics_filename), itertools.repeat(train_data), itertools.repeat(test_data), model_params_array))
    # pool_map(model_itf.generate_model, model_metrics_arg)
    # else:

    # model_types = ModelType
    # model_types = [ModelType.K_NN]
    # model_types = [ModelType.RandomForest]
    model_types = [ModelType.XGBoost, ModelType.RandomForest,
                   ModelType.NeuralNetwork]  # , ModelType.K_NN]

    model_generator = ModelGenerator(
        sub_dir_path, train_data, test_data)

    model_l = []
    for model_type in model_types:
        for model_prefix in make_model_prefix(model_type):

            model_generator.start_model_type(
                model_type, model_prefix, bMetrics)

            model_params_array = make_model_params(model_type, model_prefix)

            best_ll = sys.float_info.max
            for model_params in model_params_array:

                model, model_dict = model_generator.generate_model(
                    model_params, model_debug)

                log_loss, _ = model_dict['log_loss'], model_dict['accuracy_score']

                if bSaveModel and (log_loss < best_ll):
                    best_ll = log_loss
                    filepath, configpath = model.save_model()
                    model_dict['model_filepath'] = filepath
                    model_dict['config_filepath'] = configpath
                    model_l.append(model_dict)

    return sub_dirname, model_l


def main():

    dirname = 'data_subsets_036'
    _, subdirs, _ = next(os.walk(dirname))
    subdirs = [subdir for subdir in subdirs if 'data_subset_' in subdir]
    print("subdirs: ", subdirs)

    bDebug = False
    bMetrics = False
    bSaveModel = True

    bMultiProc = True
    bSaveModelDict = True

    start_time = time.time()

    # Seems there is a pb with multiprocess (mult. proc w/ same dataframe?)
    model_dict_l = {}
    if bMultiProc:
        models_build_arg = list(zip(itertools.repeat(dirname), subdirs, itertools.repeat(
            bSaveModel), itertools.repeat(bMetrics), itertools.repeat(bDebug)))
        sub_dir_model_l = pool_map(
            subset_model_build, models_build_arg, collect=True, arg_tuple=True)

        model_dict_l = dict(sub_dir_model_l)
    else:
        for sub_dir in subdirs:
            _, model_dict_sub_l = subset_model_build(
                dirname, sub_dir, bSaveModel, bMetrics, bDebug)

            model_dict_l[sub_dir] = model_dict_sub_l

    print("model building done")
    print("--- %s seconds ---" % (time.time() - start_time))

    if bSaveModel or bSaveModelDict:
        subset_distrib_filepath = dirname + '/fst_layer_distribution.json'

        subset_dict = load_json(subset_distrib_filepath)

        for sub_name, subset_desc in subset_dict["subsets"].items():

            if sub_name in model_dict_l.keys():
                subset_desc['models'] = model_dict_l[sub_name]

        with open(subset_distrib_filepath, 'w') as fp:
            json.dump(subset_dict, fp, indent=4)


if __name__ == '__main__':
    main()

import os
import json
import errno

from ..strat import StratConstitution
from ..common import *

from .generate_models import load_data_by_eras, load_valid_cl_data
from .model_generator import ModelGenerator
from .model_abstract import Model, ModelType
from .model_params import interpo_params

CL_INTERPO_FP = 'interpo_fp.json'


def cl_model_build(strat_dir,
                   cluster,
                   cl_dict,
                   bMetrics=False,
                   model_debug=False,
                   save=False):
    cl_dirpath = strat_dir + '/' + cluster

    cl_eras = cl_dict['eras_name']

    cl_fts = cl_dict['selected_features']
    cl_cols = ['id', 'era'] + cl_fts + ['target']

    train_data = load_data_by_eras(TRAINING_DATA_FP, cl_eras, cols=cl_cols)

    # Need to reorder ft col by selected_features in model description
    cl_order_col = ['era'] + cl_fts + ['target']
    train_data = train_data[cl_order_col]

    cl_valid_data = load_valid_cl_data(TOURNAMENT_DATA_FP, [VALID_TYPE],
                                       cl_cols)

    model_generator = ModelGenerator(cl_dirpath)
    train_input, train_target = model_generator.format_train_data(train_data)

    interpo_params_array = interpo_params()

    cl_interpo_d = dict()

    for ft in cl_fts:
        train_ft_data = train_input[ft]
        print('cl_valid_data: ', cl_valid_data)
        valid_ft_data = cl_valid_data[[ft, 'target']]

        model_generator.start_model_type(ModelType.UnivPolyInterpo, ft,
                                         bMetrics)

        for i_params in interpo_params_array:

            print("i_params: ", i_params)

            model_generator.generate_model(i_params, model_debug)
            model = model_generator.build_model(train_ft_data, train_target)

            interpo_d = dict()

            # print(" === evaluation - test data ===")
            # test_eval = model_generator.evaluate_model(test_data)
            print(" === tourn. validation - test data ===")
            valid_eval = model_generator.evaluate_model(
                valid_ft_data, cl_dirpath)

            log_loss = valid_eval['log_loss']

            if log_loss < best_ll:
                best_ll = log_loss
                filepath, configpath = model.save_model()
                # model_dict['test_eval'] = test_eval
                interpo_d['valid_eval'] = valid_eval
                interpo_d['model_filepath'] = filepath
                interpo_d['config_filepath'] = configpath

        interpo_name = model.model_name
        cl_interpo_d[interpo_name] = interpo_d

    cl_interpo_fp = cl_dirpath + '/' + CL_INTERPO_FP
    with open(cl_interpo_fp, 'w') as fp:
        json.dump(cl_interpo_d, fp)


def generate_cl_interpo(strat_dir, metrics, debug, save, cluster=None):
    strat_c_fp = strat_dir + '/' + STRAT_CONSTITUTION_FILENAME

    strat_c = StratConstitution(strat_c_fp)
    strat_c.load()

    if cluster is not None:
        cl_interpo_dir = strat_dir + '/' + cluster + '/' + INTERPO_DIRNAME
        try:
            os.makedirs(cl_interpo_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print(f"error creating dir {cl_interpo_dir}")
                exit(1)

        cl_dict = strat_c.clusters[cluster]
        cl_model_build(strat_dir, cluster, cl_dict, metrics, debug, save)
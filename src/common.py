ERA_CL_DIRNAME = 'data_clusters'
ERA_GRAPH_DIRNAME = 'data_graph'
ERA_CL_SUBDIR_PREFIX = 'cluster_'
ERA_GRAPH_SUBDIR_PREFIX = 'graph_'
DATA_DIRNAME = 'data'
SND_LAYER_DIRNAME = 'snd_layer'

STRAT_CLUSTER = 'cluster'
STRAT_ERA_GRAPH = 'era_graph'

TRAINING_DATA_FP = DATA_DIRNAME + '/numerai_training_data.csv'
ERA_LABEL = 'era'

TOURNAMENT_NAME = 'kazutsugi'
PREDICTION_NAME = 'prediction_kazutsugi'
TARGET_LABEL = f"target_{TOURNAMENT_NAME}"
TARGET_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
TARGET_CLASSES = ['0.0', '0.25', '0.5', '0.75', '1.0']
TARGET_FACT_NUMERIC = 4
CORR_THRESHOLD = 0.036
COL_PROBA_NAMES = ['proba_0.0', 'proba_0.25',
                   'proba_0.5', 'proba_0.75', 'proba_1.0']
LAYER_PRED_SUFFIX = {1: '_fst', 2: '_snd'}

FST_LAYER_TRAIN_RATIO = 0.20
TEST_RATIO = 0.20

# era_ft_corr
ERAS_FT_T_CORR_FP = DATA_DIRNAME + '/eras_ft_target_corr.csv'

# clustering
MIN_NUM_ERAS = 6
# MIN_NUM_ERAS = 3
MODEL_CONSTITUTION_FILENAME = 'model_constitution.json'
ERA_CROSS_SCORE_FP = ERA_CL_DIRNAME + '/era_cross_score.csv'

DATA_LAYER_FILENAME = 'numerai_training_data_layer.csv'
FST_LAYER_DISTRIB_FILENAME = 'numerai_training_data_layer.csv'
CL_NUMERAI_TR_DATA_FP = 'numerai_training_data.csv'

SND_LAYER_FILENAME = 'snd_layer_training_data.csv'

TOURNAMENT_DATA_FP = DATA_DIRNAME + '/numerai_tournament_data.csv'
PREDICTIONS_FILENAME = 'predictions_tournament_'
FINAL_PREDICT_FILENAME = 'final_predict_'
FINAL_PRED_VALID_FILENAME = 'final_predict_validation_'
PRED_FST_SUFFIX = '_fst_layer.csv'
PRED_SND_SUFFIX = '_snd_layer.csv'

PREDICTION_TYPES = ['validation', 'test', 'live']

ERA_CL_DIRNAME = 'data_clusters'
ERA_CL_DIRNAME_2 = 'data_clusters_2'
ERA_GRAPH_DIRNAME = 'data_graph'
ERA_SIMPLE_DIRNAME = 'data_eras'
ERA_CL_SUBDIR_PREFIX = 'cluster_'
ERA_GRAPH_SUBDIR_PREFIX = 'graph_'
DATA_DIRNAME = 'data'
SND_LAYER_DIRNAME = 'snd_layer'
INTERPO_DIRNAME = 'interpo'

STRAT_CLUSTER = 'cluster'
STRAT_CLUSTER_2 = 'cluster_2'
STRAT_ERA_GRAPH = 'era_graph'
STRAT_ERA_SIMPLE = 'era'
CLUSTER_METHODS = [
    STRAT_CLUSTER, STRAT_CLUSTER_2, STRAT_ERA_GRAPH, STRAT_ERA_SIMPLE
]
PRED_OPERATIONS = ['proba', 'prediction', 'upload']
STRA_DIRNAME_DICT = {
    STRAT_CLUSTER: ERA_CL_DIRNAME,
    STRAT_CLUSTER_2: ERA_CL_DIRNAME_2,
    STRAT_ERA_SIMPLE: ERA_SIMPLE_DIRNAME,
    STRAT_ERA_GRAPH: ERA_GRAPH_DIRNAME
}
STRA_CL_PREFIX_DIR_DICT = {
    STRAT_CLUSTER: 'cluster_',
    STRAT_ERA_SIMPLE: 'era',
    STRAT_ERA_GRAPH: 'cluster_'
}

SELECTED_FT_MUT_I = 'selected_ft_mut_i_matrix.csv'
CL_SELECTED_FT_JSON = DATA_DIRNAME + '/selected_ft.json'

TRAINING_DATA_FP = DATA_DIRNAME + '/numerai_training_data.csv'
TRAINING_STORE_H5_FP = DATA_DIRNAME + '/training_store.h5'
ERA_LABEL = 'era'

TOURNAMENT_NAME = 'kazutsugi'
PREDICTION_NAME = 'prediction_kazutsugi'
# TARGET_LABEL = f"target_{TOURNAMENT_NAME}"
TARGET_LABEL = "target"
TARGET_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
TARGET_CLASSES = ['0.0', '0.25', '0.5', '0.75', '1.0']
CLASS_WEIGHT = [0.04, 0.19, 0.5, 0.19, 0.04]
SCALE_POS_WEIGHT = 2
TARGET_FACT_NUMERIC = 4
CORR_THRESHOLD = 0.036
COL_PROBA_NAMES = [
    'proba_0.0', 'proba_0.25', 'proba_0.5', 'proba_0.75', 'proba_1.0'
]

ERAS_FT_TARGET_MI_FP = DATA_DIRNAME + '/eras_features_target_mi.csv'
MI_MAT_FP = DATA_DIRNAME + '/mut_i_mat.csv'

FST_LAYER_TRAIN_RATIO = 0.20
TEST_RATIO = 0.10

# era_ft_corr
ERAS_FT_T_CORR_FP = DATA_DIRNAME + '/eras_ft_target_corr.csv'

STRAT_CONSTITUTION_FILENAME = 'strat_constitution.json'
MODEL_AGGREGATION_FILENAME = 'model_aggregations.json'
ERA_CROSS_SCORE_FP = ERA_CL_DIRNAME + '/era_cross_score.csv'

DATA_LAYER_FILENAME = 'numerai_training_data_layer.csv'
FST_LAYER_DISTRIB_FILENAME = 'numerai_training_data_layer.csv'
CL_NUMERAI_TR_DATA_FP = 'numerai_training_data.csv'

FST_LAYER = 'fst_layer'
SND_LAYER = 'snd_layer'
LAYERS = ['0', 'fst', 'layer']
LAYER_PRED_SUFFIX = {FST_LAYER: '_fst.csv', SND_LAYER: '_snd.csv'}

SND_LAYER_FILENAME = 'snd_layer_training_data.csv'

TOURNAMENT_DATA_FP = DATA_DIRNAME + '/numerai_tournament_data.csv'
TOURNAMENT_STORE_H5_FP = DATA_DIRNAME + '/tournament_store.h5'
PREDICTIONS_FILENAME = 'predictions_tournament_'
PRED_TRAIN_FILENAME = 'predictions_training_data.csv'
PROBA_FILENAME = 'proba_tournament_'
FINAL_PREDICT_FILENAME = 'final_predict_'
FINAL_PRED_VALID_FILENAME = 'final_predict_validation'
FINAL_PRED_LIVE_FILENAME = 'final_predict_live'
FINAL_PRED_TEST_FILENAME = 'final_predict_test'
PRED_FST_SUFFIX = '_fst.csv'
PRED_SND_SUFFIX = '_snd.csv'

TRAINING_TYPE = 'training'
VALID_TYPE = 'validation'
TEST_TYPE = 'test'
LIVE_TYPE = 'live'
PREDICTION_TYPES = [VALID_TYPE, TEST_TYPE, LIVE_TYPE]

H5_ERAS = 'eras'
H5_FT = 'features'

NUMERAI_PRED_FILENAME = "numerai_prediction.csv"
PREDICT_LABEL = 'prediction'

COMPUTE_BOOL = False

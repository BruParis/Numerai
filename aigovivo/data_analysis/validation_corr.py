import pandas as pd
import numpy as np

from ..common import *
from ..reader import ReaderCSV

from .feature_era_corr import compute_corr

BENCHMARK = 0
BAND = 0.2


def load_target():
    file_reader = ReaderCSV(TOURNAMENT_DATA_FP)
    input_data = file_reader.read_csv_matching('data_type', [VALID_TYPE],
                                               columns=['id', 'era', 'target'
                                                        ]).set_index('id')

    return input_data


def payout(scores):
    return ((scores - BENCHMARK) / BAND).clip(lower=-1, upper=1)


def score(data_df, pred_col_name):
    # Submissions are scored by spearman correlation
    # method="first" breaks ties based on order in array
    res = compute_corr(data_df[TARGET_LABEL], data_df[pred_col_name])
    return res


def compute_score(data_df, pred_col_name, bDebug=False):

    if pred_col_name not in data_df.columns:
        return

    if bDebug:
        print("Validation score for prediction type: ", pred_col_name)

    data_correlations = data_df.groupby("era").apply(score, pred_col_name)

    corr_mean = data_correlations.mean()
    corr_std = data_correlations.std()
    corr_sharpe = corr_mean / corr_std
    # payout = payout(data_correlations).mean()

    if bDebug:
        print(
            f"On data the correlation has mean {corr_mean} and std {corr_std}")
    # print(f"On data the average per-era payout is {valid_payout}")

    data_score = {
        'corr_mean': corr_mean,
        'corr_std': corr_std,
        'corr_sharpe': corr_sharpe
    }

    # Check consistency -> need proba
    # print("=============================")
    # eras = validation_data.era.unique()

    # good_eras = 0

    # for era in eras:
    #    tmp = validation_data[validation_data['era'] == era]
    #    ll = log_loss(tmp[TARGET_NAME], tmp[pred_col_name])
    #    is_good = ll < -log(0.5)

    #    if is_good:
    #        good_eras += 1

    #    print("{} {} {:.2%} {}".format(era, len(tmp), ll, is_good))

    # consistency = good_eras / float(len(eras))
    # print(
    #    "\nconsistency: {:.1%} ({}/{})".format(consistency, good_eras, len(eras)))

    # ll = log_loss(validation_data.target, validation_data.probability)
    # print("log loss:    {:.2%}\n".format(ll))

    # res = {'validation_corr': validation_corr_desc,
    #       'consistency: ': consistency, 'log_loss': ll}

    return data_score


def models_valid_score(models_dict, model_types, pred_models_df):
    data_target_df = load_target()
    print('target_df: ', data_target_df)

    data_target_df = pd.concat([data_target_df, pred_models_df], axis=1)

    print('data_target_df: ', data_target_df)
    print('model_types: ', model_types)

    models_scores = {
        model.name: compute_score(data_target_df, model.name)
        for model in model_types
    }

    print('models_scores: ', models_scores)
    print('models_dict: ', models_dict)
    for model, valid_scorr in models_scores.items():
        models_dict[model] = valid_scorr


def pred_score(pred_f, pred_name, data_target, bDebug=True):

    data_df = pd.concat([data_target, pred_f], axis=1)
    res = compute_score(data_df, pred_name, bDebug)

    return res

import pandas as pd
import numpy as np
from common import *
from reader import ReaderCSV

BENCHMARK = 0
BAND = 0.2

# move to corr_analysis, make it more generic tool


def load_validation_target():
    file_reader = ReaderCSV(TOURNAMENT_DATA_FP)
    input_data = file_reader.read_csv_matching(
        'data_type', [VALID_TYPE], columns=['id', 'era', 'target']).set_index('id')

    return input_data


def payout(scores):
    return ((scores - BENCHMARK) / BAND).clip(lower=-1, upper=1)


def score(data_df, pred_col_name):
    # Submissions are scored by spearman correlation
    # method="first" breaks ties based on order in array
    return np.corrcoef(
        data_df[TARGET_LABEL],
        data_df[pred_col_name].rank(pct=True, method="first")
    )[0, 1]


def pred_valid_score(validation_data, pred_col_name):

    if pred_col_name not in validation_data.columns:
        return

    print("Validation score for prediction type: ", pred_col_name)

    print("validation_data: ", validation_data)

    validation_correlations = validation_data.groupby(
        "era").apply(score, pred_col_name)

    valid_corr_mean = validation_correlations.mean()
    valid_corr_std = validation_correlations.std()
    valid_payout = payout(validation_correlations).mean()

    print(
        f"On validation the correlation has mean {valid_corr_mean} and std {valid_corr_std}")
    print(
        f"On validation the average per-era payout is {valid_payout}")

    valid_score = {'valid_corr_mean': valid_corr_mean,
                   'valid_corr_std': valid_corr_std, 'valid_payout': valid_payout}

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

    return valid_score


def models_valid_score(models_dict, model_types, pred_models_df):
    valid_df = load_validation_target()

    valid_df = pd.concat([valid_df, pred_models_df], axis=1)

    print('model_types: ', model_types)

    models_scores = {model.name: pred_valid_score(
        valid_df, model.name) for model in model_types}

    print('models_scores: ', models_scores)
    print('models_dict: ', models_dict)
    for model, valid_scorr in models_scores.items():
        models_dict[model]['valid_score'] = valid_scorr


def valid_score(pred_f, pred_name):
    valid_df = load_validation_target()

    valid_df = pd.concat([valid_df, pred_f], axis=1)
    res = pred_valid_score(valid_df, pred_name)

    return res

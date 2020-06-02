import sys
import json
import numpy as np
import pandas as pd
from math import log
from sklearn.metrics import log_loss

from reader_csv import ReaderCSV

TOURNAMENT_NAME = "kazutsugi"
TARGET_NAME = f"target_{TOURNAMENT_NAME}"
PREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"

BENCHMARK = 0
BAND = 0.2

try:
    validation_pred_filepath = sys.argv[1]
except IndexError:
    print("Usage: validation_score.py <final_prediction_validation.csv>")
    print("-----> default value : data_subsets_036/final_predict_validation_snd.csv")
    validation_pred_filepath = 'data_subsets_036/final_predict_validation_snd.csv'


def score(data_df, pred_name):
    # Submissions are scored by spearman correlation
    # method="first" breaks ties based on order in array
    return np.corrcoef(
        data_df[TARGET_NAME],
        data_df[pred_name].rank(pct=True, method="first")
    )[0, 1]


# The payout function
def payout(scores):
    return ((scores - BENCHMARK) / BAND).clip(lower=-1, upper=1)


def load_validation_data():
    data_filepath = 'numerai_tournament_data.csv'
    file_reader = ReaderCSV(data_filepath)
    validation_data = file_reader.read_csv_matching(
        'data_type', ['validation'], columns=['id', 'era', TARGET_NAME]).set_index('id')

    return validation_data


def load_validation_prediction(pred_validation_filepath):
    file_reader = ReaderCSV(pred_validation_filepath)
    pred_data = file_reader.read_csv().set_index('id')

    return pred_data


def pred_type_score(validation_data, pred_name):

    print("Validation score for prediction type: ",)

    validation_correlations = validation_data.groupby(
        "era").apply(score, pred_name)

    valid_corr_mean = validation_correlations.mean()
    valid_corr_std = validation_correlations.std()
    valid_payout = payout(validation_correlations).mean()

    print(
        f"On validation the correlation has mean {valid_corr_mean} and std {valid_corr_std}")
    print(
        f"On validation the average per-era payout is {valid_payout}")

    validation_corr_desc = {'valid_corr_mean': valid_corr_mean,
                            'valid_corr_std': valid_corr_std, 'valid_payout': valid_payout}

    # Check consistency
    print("=============================")
    eras = validation_data.era.unique()

    good_eras = 0

    for era in eras:
        tmp = validation_data[validation_data['era'] == era]
        ll = log_loss(tmp[TARGET_NAME], tmp[pred_name])
        is_good = ll < -log(0.5)

        if is_good:
            good_eras += 1

        print("{} {} {:.2%} {}".format(era, len(tmp), ll, is_good))

    consistency = good_eras / float(len(eras))
    print(
        "\nconsistency: {:.1%} ({}/{})".format(consistency, good_eras, len(eras)))

    ll = log_loss(validation_data.target, validation_data.probability)
    print("log loss:    {:.2%}\n".format(ll))

    res = {'validation_corr': validation_corr_desc,
           'consistency: ': consistency, 'log_loss': ll}

    return res


def main():

    # Check the per-era correlations on the validation set
    validation_data = load_validation_data()
    pred_data = load_validation_prediction(validation_pred_filepath)

    validation_data = pd.concat(
        [validation_data, pred_data], axis=1)

    pred_score = {pred_name: pred_type_score(validation_data, pred_name) for pred_name in [
        'xgboost', 'rf', 'neural_net', 'arithmetic_mean']}

    pred_score_filepath = validation_pred_filepath.replace(
        '.csv', '_score.json')
    with open(pred_score_filepath, 'w') as fp:
        json.dump(pred_score, fp, indent=4)


if __name__ == '__main__':
    main()

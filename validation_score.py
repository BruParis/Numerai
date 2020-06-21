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
    layer_pred = sys.argv[1]
except IndexError:
    print(
        "Usage: validation_score.py <layer used for pred : ['full', 'fst', 'snd']>")
    print("-----> default value : snd")
    layer_pred = 'snd'


def load_json(filepath):
    with open(filepath, 'r') as f:
        json_data = json.load(f)

        return json_data


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

    if 'era' in pred_data.columns:
        pred_data = pred_data.drop('era', axis=1)

    return pred_data


def pred_type_score(validation_data, pred_name):

    if pred_name not in validation_data.columns:
        return

    print("Validation score for prediction type: ", pred_name)

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

    # Check consistency -> need proba
    # print("=============================")
    # eras = validation_data.era.unique()

    # good_eras = 0

    # for era in eras:
    #    tmp = validation_data[validation_data['era'] == era]
    #    ll = log_loss(tmp[TARGET_NAME], tmp[pred_name])
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

    res = {'validation_corr': validation_corr_desc}

    return res


def main():

    dirname = 'data_subsets_036'

    final_pred_descr_fp = dirname + '/final_predict_scores.json'
    final_pred_descr = load_json(final_pred_descr_fp)

    validation_pred_filepath = {'full': dirname + '/final_predict_validation_full.csv',
                                'fst': dirname + '/final_predict_validation_fst.csv',
                                'snd': dirname + '/final_predict_validation_snd.csv'}

    layer_pred_filepath = validation_pred_filepath[layer_pred]

    pred_types = final_pred_descr[layer_pred]['models'] + ['arithmetic_mean']

    # Check the per-era correlations on the validation set
    validation_data = load_validation_data()
    pred_data = load_validation_prediction(layer_pred_filepath)

    validation_data = pd.concat(
        [validation_data, pred_data], axis=1)

    pred_scores = {pred_name: pred_type_score(
        validation_data, pred_name) for pred_name in pred_types}

    final_pred_descr[layer_pred]['scores'] = pred_scores

    with open(final_pred_descr_fp, 'w') as fp:
        json.dump(final_pred_descr, fp, indent=4)


if __name__ == '__main__':
    main()

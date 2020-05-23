import numpy as np
import pandas as pd

from reader_csv import ReaderCSV

TOURNAMENT_NAME = "kazutsugi"
TARGET_NAME = f"target_{TOURNAMENT_NAME}"
PREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"

BENCHMARK = 0
BAND = 0.2


def score(data_df):
    # Submissions are scored by spearman correlation
    # method="first" breaks ties based on order in array
    return np.corrcoef(
        data_df[TARGET_NAME],
        data_df[PREDICTION_NAME].rank(pct=True, method="first")
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
    pred_data = file_reader.read_csv(
        columns=['id', PREDICTION_NAME]).set_index('id')

    return pred_data


def main():

    validation_pred_filepath = 'data_subsets_036/predictions_tournament_validation.csv'
    validation_pred_data = load_validation_prediction(validation_pred_filepath)

    # Check the per-era correlations on the validation set
    validation_data = load_validation_data()

    print("validation_data: ", validation_data)
    print("validation_pred_data: ", validation_pred_data)

    validation_data = pd.concat(
        [validation_data, validation_pred_data], axis=1)
    print("validation_data: ", validation_data)

    validation_correlations = validation_data.groupby("era").apply(score)
    print(
        f"On validation the correlation has mean {validation_correlations.mean()} and std {validation_correlations.std()}")
    print(
        f"On validation the average per-era payout is {payout(validation_correlations).mean()}")


if __name__ == '__main__':
    main()

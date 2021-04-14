import pandas as pd
import numpy as np


def neutralize(data_df, pred_col_name, by=None, proportion=1.0):
    if by is None:
        by = [x for x in data_df.columns if x.startswith('feature')]

    scores = data_df[pred_col_name].copy()
    exposures = data_df[by].values

    # constant column to make sure the series is completely neutral to exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(scores)] * len(exposures)).reshape(-1, 1)))

    scores -= proportion * (
        exposures @ (np.linalg.pinv(exposures) @ scores.values))

    res = scores / scores.std()

    return res

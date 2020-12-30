import pandas as pd

from common import *
from models import ModelType


def proba_to_target_label(proba, models):
    res = pd.DataFrame()
    for eModel in models:

        model_name = eModel.name
        model_proba = proba.loc[:, proba.columns.str.startswith(model_name)]

        if model_proba is None:
            continue

        proba_to_target_classes = dict(
            zip(model_proba.columns, TARGET_CLASSES))
        model_proba = model_proba.rename(columns=proba_to_target_classes)

        pred_df = model_proba.idxmax(axis=1)
        pred_df = pred_df.rename(eModel.name)

        res = pd.concat([res, pred_df], axis=1)

    if 'era' in proba.columns:
        res = pd.concat([res, proba.loc[:, 'era']], axis=1)

    return res


def rank_proba(proba, models):
    res = pd.DataFrame()
    for eModel in models:

        model_name = eModel.name
        model_proba = proba.loc[:, proba.columns.str.startswith(model_name)]

        if model_proba is None:
            continue

        score_df = pd.concat([model_proba.loc[:, model_proba.columns.str.endswith(
            str(t_val))] * t_val for t_val in TARGET_VALUES], axis=1).sum(axis=1)
        rank_s = score_df.rank() / len(score_df)
        rank_df = pd.DataFrame(rank_s, columns=[model_name])

        res = pd.concat([res, rank_df], axis=1)

    if 'era' in proba.columns:
        res = pd.concat([res, proba.loc[:, 'era']], axis=1)

    return res


def rank_pred(pred_df):
    rank_s = pred_df.rank() / len(pred_df)

    return rank_s

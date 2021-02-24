import pandas as pd

from ..common import *


def proba_to_target_label(proba, models):
    res = pd.DataFrame()
    for eModel in models:

        model_name = eModel.name
        model_proba = proba.loc[:, proba.columns.str.startswith(model_name)]

        if model_proba is None:
            continue

        proba_to_target_classes = dict(zip(model_proba.columns,
                                           TARGET_CLASSES))
        model_proba = model_proba.rename(columns=proba_to_target_classes)

        pred_df = model_proba.idxmax(axis=1)
        pred_df = pred_df.rename(eModel.name)

        res = pd.concat([res, pred_df], axis=1)

    if 'era' in proba.columns:
        res = pd.concat([res, proba.loc[:, 'era']], axis=1)

    return res


def rank_pred(pred_df):
    rank_s = pred_df.rank(pct=True, method="first")

    return rank_s


def rank_proba(proba, model_name):
    score_df = pd.concat([
        proba.loc[:, proba.columns.str.endswith(str(t_val))] * t_val
        for t_val in TARGET_VALUES
    ],
                         axis=1).sum(axis=1)
    rank_s = rank_pred(score_df)
    rank_df = pd.DataFrame(rank_s, columns=[model_name])

    if 'era' in proba.columns:
        rank_df = pd.concat([rank_df, proba.loc[:, 'era']], axis=1)

    return rank_df


def rank_proba_models(proba, models):
    res = pd.DataFrame()
    for eModel in models:

        model_name = eModel.name
        model_proba = proba.loc[:, proba.columns.str.startswith(model_name)]

        if model_proba is None:
            continue

        model_rank_df = rank_proba(model_proba, model_name)
        res = pd.concat([res, model_rank_df], axis=1)

    if 'era' in proba.columns:
        res = pd.concat([res, proba.loc[:, 'era']], axis=1)

    return res

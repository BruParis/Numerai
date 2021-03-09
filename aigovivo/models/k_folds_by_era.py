import random
import itertools


def cv_k_folds_by_era(data_df, num_folds=5):
    k_folds = {k: [] for k in range(num_folds)}
    for _, era_idx in data_df.groupby('era').apply(
            lambda x: x.index.to_numpy()).items():
        random.shuffle(era_idx)

        fold_len = int(len(era_idx) / num_folds)
        for k in range(num_folds):
            k_folds[k] += list(era_idx[k * fold_len:(k + 1) * fold_len])

    res = []
    for k in range(num_folds):
        train_split_idx = list(
            itertools.chain.from_iterable(
                [v for j, v in k_folds.items() if j != k]))
        res.append((train_split_idx, k_folds[k]))

    return res

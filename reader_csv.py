import pandas as pd
import csv
import os
import numpy as np


class ReaderCSV():

    def _get_dtypes(self):
        #print("get dtypes")
        with open(self.filepath, 'r') as f:
            column_names = next(csv.reader(f))
            dtypes = {x: self.data_type for x in column_names if
                      x.startswith(('feature', 'target'))}

        return dtypes

    def __init__(self, filepath, data_type=np.float32):

        # try:
        #    os.path.isfile(filepath):
        # except OSError as e:
        #    print("")
        #    return

        self.filepath = filepath
        self.data_type = data_type

    def read_csv_header(self):
        with open(self.filepath, 'r') as f:
            column_names = next(csv.reader(f))

        return pd.DataFrame(columns=column_names)

    def read_csv_matching(self, filter_col, filter_values, columns=None):
        dtypes = self._get_dtypes()

        print("loading: ", self.filepath)

        data_col = pd.read_csv(self.filepath, usecols=[filter_col])
        data_col_fst = data_col.drop_duplicates()
        data_col_last = data_col.drop_duplicates(keep='last')

        data_col_fst_idx = data_col_fst.loc[data_col[filter_col].isin(
            filter_values)].index.values
        data_col_last_idx = data_col_last.loc[data_col[filter_col].isin(
            filter_values)].index.values

        data_fst_last_idx = zip(data_col_fst_idx, data_col_last_idx)

        data_match = [pd.read_csv(self.filepath,
                                  skiprows=lambda x: (
                                      x <= fst_idx and x != 0) or x > last_idx,
                                  usecols=columns,
                                  dtype=dtypes)
                      for fst_idx, last_idx in data_fst_last_idx]

        data_match_full = pd.concat(data_match)
        return data_match_full

    def read_csv(self, skip_lambda=None, columns=None):
        # Read the csv file into a pandas Dataframe
        print("loading: ", self.filepath)

        dtypes = self._get_dtypes()
        #print("read csv")
        return pd.read_csv(self.filepath, skiprows=skip_lambda, usecols=columns, dtype=dtypes)

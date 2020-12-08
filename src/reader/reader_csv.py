import pandas as pd
import csv
import os
import numpy as np


class ReaderCSV():

    def _get_dtypes(self):
        # print("get dtypes")
        with open(self.filepath, 'r') as f:
            column_names = next(csv.reader(f))
            dtypes = {x: self.data_type for x in column_names if
                      x.startswith(('feature', 'target'))}

        return dtypes

    # Load with dask and convert to pandas
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

    def read_csv_filter_id(self, list_id, columns=None):
        dtypes = self._get_dtypes()
        print("loading: ", self.filepath)

        # Do not use skiprows with filter_id -> usually list_id can pretty long
        #    (each row reading will look into list_id)
        data_df = pd.read_csv(self.filepath, usecols=columns, dtype=dtypes)
        data_filtered_df = data_df[data_df['id'].isin(list_id)]

        # print("read csv")
        return data_filtered_df

    def read_idx_to_idx(self, fst_idx, last_idx, columns=None):
        data_df = pd.read_csv(self.filepath,
                              skiprows=lambda x: (
                                  x <= fst_idx and x != 0) or x > last_idx,
                              usecols=columns)

        return data_df

    def read_csv_matching(self, filter_col, filter_values, columns=None):

        full_data_id = pd.read_csv(self.filepath, usecols=['id'])
        data_col = pd.read_csv(self.filepath, usecols=['id', filter_col])
        data_col_sel = data_col.loc[data_col[filter_col].isin(
            filter_values)]

        data_skip_ids = full_data_id.drop(data_col_sel.index)
        data_skip_ids = np.array(data_skip_ids.dropna().index) + 1
        data_skip_ids = np.delete(
            data_skip_ids, np.where(data_skip_ids == 1), axis=0)

        data_match = pd.read_csv(self.filepath,
                                 skiprows=data_skip_ids,
                                 usecols=columns)

        return data_match

    def read_csv(self, skip_lambda=None, columns=None):
        # Read the csv file into a pandas Dataframe
        print("loading: ", self.filepath)

        dtypes = self._get_dtypes()
        # print("read csv")
        data_df = pd.read_csv(
            self.filepath, skiprows=skip_lambda, usecols=columns, dtype=dtypes)
        return data_df

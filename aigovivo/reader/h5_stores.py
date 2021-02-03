import pandas as pd
import os
from ..reader import ReaderCSV
from ..common import *


def load_h5_eras(data_fp, eras):
    print("eras: ", eras)
    data_df_l = [pd.read_hdf(data_fp, e) for e in eras]
    res = pd.concat(data_df_l, axis=0)

    return res

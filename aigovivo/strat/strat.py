import os
import errno

from ..common import *
from .strat_constitution import StratConstitution

CL_MIN_NUM_ERAS = 3
CL_2_MIN_NUM_ERAS = 8


def make_new_strat(method, strat_dir):
    print("make new strat dir: ", strat_dir)
    try:
        os.makedirs(strat_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print("a dir for {} already exists! ", strat_dir)
            exit(1)

    strat_c_fp = strat_dir + '/' + STRAT_CONSTITUTION_FILENAME
    strat_c = StratConstitution(strat_c_fp)

    strat_c.eras_ft_t_corr_file = ERAS_FT_T_CORR_FP

    min_num_eras = -1
    if method == STRAT_CLUSTER:
        min_num_eras = CL_MIN_NUM_ERAS
    elif method == STRAT_CLUSTER_2:
        min_num_eras = CL_2_MIN_NUM_ERAS

    strat_c.cl_params = {'method': method, 'min_num_eras': min_num_eras}
    strat_c.save()

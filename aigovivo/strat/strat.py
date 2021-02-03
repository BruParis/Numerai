import os
import errno

from ..common import *
from .model_constitution import ModelConstitution


def make_new_strat(strat_fp):
    print("make new strat dir: ", strat_fp)
    try:
        os.makedirs(strat_fp)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print("a dir for {} already exists! ", strat_fp)
            exit(1)

    cl_c_filename = strat_fp + '/model_constitution.json'
    cl_c = ModelConstitution(cl_c_filename)
    cl_c.eras_ft_t_corr_file = ERAS_FT_T_CORR_FP
    cl_c.save()

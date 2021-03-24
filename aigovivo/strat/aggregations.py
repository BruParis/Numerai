import json
import itertools

from ..common import *

from ..models import ModelDescription

from .strat_constitution import StratConstitution


class Aggregations():
    def __init__(self, filepath):
        self.filepath = filepath
        self.aggr_dict = None

    def _gen_aggr_dict(self, layer, cl_dict):

        model_types = [
            ModelType.XGBoost, ModelType.RandomForest, ModelType.NeuralNetwork
        ]

        model_combs = []
        for r in range(len(model_types)):
            r_comb = itertools.combinations(model_types, r + 1)
            for comb in r_comb:
                model_combs.append(comb)

        self.aggr_dict = dict()
        for m_c in model_combs:
            m_c_w_sorted = []
            m_c_found = set()

            for eModel in m_c:
                model_name = eModel.name
                for cl, cl_desc in cl_dict.items():

                    cl_desc_m = cl_desc['models']
                    if model_name not in cl_desc_m.keys():
                        continue

                    m_c_found.add(eModel)

                    model_fp = cl_desc_m[model_name]
                    print("model_fp: ", model_fp)
                    model_desc = ModelDescription(layer, model_fp, eModel)
                    model_desc.load()

                    cl_m_w = model_desc.valid_eval['eval_score']['corr_mean']
                    if cl_m_w > 0:
                        m_c_w_sorted.append((cl, model_name, cl_m_w))

                m_c_w_sorted = sorted(m_c_w_sorted,
                                      key=lambda x: x[2],
                                      reverse=True)

            m_aggr_dict = dict()
            if set(m_c) == m_c_found:
                for i in range(1, len(m_c_w_sorted)):
                    aggr_cl_m_w = m_c_w_sorted[:i]
                    aggr_pred_name = AGGR_PREFIX + str(i)
                    m_aggr_dict[aggr_pred_name] = {
                        'cluster_models':
                        [(cl, m, w) for cl, m, w in aggr_cl_m_w],
                        'total_w': sum([w for *_, w in aggr_cl_m_w])
                    }

            m_c_key = ','.join([m.name for m in m_c])
            self.aggr_dict[m_c_key] = m_aggr_dict

    def get_models_aggr(self, model_types):

        for m_comb, m_comb_aggr in self.aggr_dict.items():
            aggr_models = set([ModelType[m] for m in m_comb.split(',')])
            if set(model_types) == aggr_models:
                return m_comb_aggr

        print("/!\ no aggr. found for this model comb: {}".format(model_types))
        return None

    def save(self):

        with open(self.filepath, 'w') as f:
            json.dump(self.aggr_dict, f, indent=4)

    def load(self):
        with open(self.filepath, 'r') as fp:
            self.aggr_dict = json.load(fp)


def make_aggr_dict(layer, folder):

    strat_c_fp = folder + '/' + STRAT_CONSTITUTION_FILENAME
    strat_c = StratConstitution(strat_c_fp)
    strat_c.load()

    cl_dict = strat_c.clusters

    model_a_filepath = folder + '/' + MODEL_AGGREGATION_FILENAME
    aggr_dict = Aggregations(model_a_filepath)
    aggr_dict._gen_aggr_dict(layer, cl_dict)
    aggr_dict.save()

    return
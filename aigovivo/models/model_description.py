import json

from ..common import ModelType, STRAT_CONSTITUTION_FILENAME
from ..models import model_params, get_desc_filename
from ..strat import StratConstitution


class ModelDescription():
    def _extract_from_dict(self, m_dict):

        self.model_type = m_dict.get('model_type')
        self.model_fp = m_dict.get('model_fp')
        self.config_fp = m_dict.get('config_fp')
        self.train_pred_fp = m_dict.get('train_pred_fp')
        self.train_params = m_dict.get('train_params')
        self.train_eval = m_dict.get('train_eval')
        self.valid_eval = m_dict.get('valid_eval')
        self.rs_params = m_dict.get('rs_params')
        self.rs_eval = m_dict.get('rs_eval')
        self.imp_fts_sel = m_dict.get('imp_fts_sel')

    def _generate_dict(self):
        model_dict = dict()

        if self.model_type is not None:
            model_dict['model_type'] = self.model_type
        if self.model_fp is not None:
            model_dict['model_fp'] = self.model_fp
        if self.config_fp is not None:
            model_dict['config_fp'] = self.config_fp
        if self.train_pred_fp is not None:
            model_dict['train_pred_fp'] = self.train_pred_fp
        if self.train_params is not None:
            model_dict['train_params'] = self.train_params
        if self.train_eval is not None:
            model_dict['train_eval'] = self.train_eval
        if self.valid_eval is not None:
            model_dict['valid_eval'] = self.valid_eval
        if self.rs_params is not None:
            model_dict['rs_params'] = self.rs_params
        if self.rs_eval is not None:
            model_dict['rs_eval'] = self.rs_eval
        if self.imp_fts_sel is not None:
            model_dict['imp_fts_sel'] = self.imp_fts_sel

        return model_dict

    def __init__(self, layer, filepath, model_type):
        self.layer = layer
        self.filepath = filepath
        self.model_type = model_type
        self.model_fp = None
        self.config_fp = None
        self.train_pred_fp = None
        self.train_params = None
        self.train_eval = None
        self.valid_eval = None
        self.rs_params = None
        self.rs_eval = None
        self.imp_fts_sel = None

    def set_default(self):
        it_model_p = model_params(self.layer, self.model_type)
        for model_p in it_model_p:
            self.train_params = model_p
            break
        return

    def save(self):
        model_c_dict = self._generate_dict()

        print("model_c_dict: ", model_c_dict)

        with open(self.filepath, 'w') as f:
            json.dump(model_c_dict, f, indent=4)

    def load(self):
        with open(self.filepath, 'r') as fp:
            loaded_dict = json.load(fp)

        self._extract_from_dict(loaded_dict)
        return


def generate_model_dict(folder, layer):
    strat_c_fp = folder + '/' + STRAT_CONSTITUTION_FILENAME
    strat_c = StratConstitution(strat_c_fp)
    strat_c.load()

    s_cl = strat_c.clusters

    models = [
        ModelType.NeuralNetwork, ModelType.XGBoost, ModelType.RandomForest
    ]

    for cl, cl_d in s_cl.items():
        cl_d['models'] = dict()

        for eModel in models:
            desc_fn = get_desc_filename(eModel)
            model_fp = folder + '/' + cl + '/' + desc_fn
            model_desc = ModelDescription(layer, model_fp, eModel)
            model_desc.set_default()

            cl_d['models'][eModel.name] = model_fp

            model_desc.save()

    strat_c.save()

    return
import json


class StratConstitution():
    def _extract_from_dict(self, m_dict):

        self.eras_ft_t_corr_file = m_dict.get('eras_ft_t_corr_file')
        self.eras_cross_score_file = m_dict.get('eras_cross_score_file')
        self.clusters = m_dict.get('clusters')
        self.cl_params = m_dict.get('cl_params')
        self.final_pred = m_dict.get('final_pred')
        self.snd_layer = m_dict.get('snd_layer')
        self.compute_aggr = m_dict.get('compute_aggr')

    def _generate_dict(self):
        strat_c_dict = dict()

        if self.eras_ft_t_corr_file is not None:
            strat_c_dict['eras_ft_t_corr_file'] = self.eras_ft_t_corr_file
        if self.eras_cross_score_file is not None:
            strat_c_dict['eras_cross_score_file'] = self.eras_cross_score_file
        if self.clusters is not None:
            strat_c_dict['clusters'] = self.clusters
        if self.cl_params is not None:
            strat_c_dict['cl_params'] = self.cl_params
        if self.final_pred is not None:
            strat_c_dict['final_pred'] = self.final_pred
        if self.snd_layer is not None:
            strat_c_dict['snd_layer'] = self.snd_layer
        if self.compute_aggr is not None:
            strat_c_dict['compute_aggr'] = self.compute_aggr

        return strat_c_dict

    def __init__(self, filepath):
        self.filepath = filepath
        self.eras_ft_t_corr_file = None
        self.eras_cross_score_file = None
        self.cl_params = None
        self.clusters = None
        self.snd_layer = None
        self.final_pred = None
        self.compute_aggr = None

    def save(self):
        model_c_dict = self._generate_dict()

        with open(self.filepath, 'w') as f:
            json.dump(model_c_dict, f, indent=4)

    def load(self):
        with open(self.filepath, 'r') as fp:
            loaded_dict = json.load(fp)

        self._extract_from_dict(loaded_dict)
        return

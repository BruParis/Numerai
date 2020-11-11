import json


class ModelConstitution():

    def _extract_from_dict(self, m_dict):

        self.orig_data_file = m_dict.get('orig_data_file')
        self.eras_ft_t_corr_file = m_dict.get('eras_ft_t_corr_file')
        self.eras_cross_score_file = m_dict.get('eras_cross_score_file')
        self.clusters = m_dict.get('clusters')
        self.prediction = m_dict.get('prediction')

    def _generate_dict(self):
        model_c_dict = dict()

        if self.orig_data_file is not None:
            model_c_dict['orig_data_file'] = self.orig_data_file
        if self.eras_ft_t_corr_file is not None:
            model_c_dict['eras_ft_t_corr_file'] = self.eras_ft_t_corr_file
        if self.eras_cross_score_file is not None:
            model_c_dict['eras_cross_score_file'] = self.eras_cross_score_file
        if self.clusters is not None:
            model_c_dict['clusters'] = self.clusters
        if self.prediction is not None:
            model_c_dict['prediction'] = self.prediction

        return model_c_dict

    def __init__(self, filepath):
        self.filepath = filepath
        self.orig_data_file = None
        self.eras_ft_t_corr_file = None
        self.eras_cross_score_file = None
        self.clusters = None
        self.prediction = None

    def save(self):
        model_c_dict = self._generate_dict()

        with open(self.filepath, 'w') as f:
            json.dump(model_c_dict, f, indent=4)

    def load(self):
        with open(self.filepath, 'r') as fp:
            loaded_dict = json.load(fp)
        self._extract_from_dict(loaded_dict)
        return

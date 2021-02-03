import numpy as np
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage


class EraFTLinkage():
    def _compute_score_proxy(self, era_1, era_2):

        if era_1 > len(self.eras):
            print('Wrong number for era_1: ', era_1)
            return None

        if era_2 > len(self.eras):
            print('Wrong number for era_2: ', era_2)
            return None

        # era_1_target_corr = self.eras_t_corr_df.loc['era' +
        #                                                  str(era_1)].abs()
        # era_2_target_corr = self.eras_t_corr_df.loc['era' +
        #                                                  str(era_2)].abs()

        era_1_s = 'era' + str(era_1)
        era_2_s = 'era' + str(era_2)

        era_12_target_corr = self.eras_t_corr_df.loc[
            (self.eras_t_corr_df['era_x'] == era_1_s)
            & (self.eras_t_corr_df['era_y'] == era_2_s)]  # .abs()

        era_12_target_corr = era_12_target_corr.drop(['era_x', 'era_y'],
                                                     axis=1).abs()
        # CHOICE
        # Use product as measure of score & proximity between eras
        # strong corr. in era i will be nullified if weak in j and
        # hence cluster strongest together, instead of using just dist.
        # which will only expose similarity

        fts_score = era_12_target_corr.sum(axis=1)
        # fts_score = abs(era_1_target_corr - era_2_target_corr)

        era_12_target_corr = era_12_target_corr.T
        self.era_i_j_ft_score[era_1 - 1][era_2 - 1] = era_12_target_corr
        # self.era_i_j_ft_score[era_1-1][era_2-1] = fts_score.sum()
        self.era_score_mat[era_1 - 1, era_2 - 1] = fts_score.sum()
        # ft_cross_corr = np.linalg.norm(era_i_t_corr - era_j_t_corr)

        return

    def _find_fst_idx_value(self, data_l, value):
        for i in range(len(data_l)):
            if data_l[i] == value:
                return i
        return -1

    def _compute_eras_proxy(self):
        for era_i in self.eras:
            self.era_i_j_ft_score[era_i - 1] = dict()
            for era_j in self.eras:
                self._compute_score_proxy(era_i, era_j)

    def __init__(self, eras_t_corr_df, data_type=np.float32):

        self.eras = [
            int(era.replace('era', ''))
            for era in set(eras_t_corr_df['era_x'].values)
        ]
        self.eras.sort()

        self.eras_t_corr_df = eras_t_corr_df
        self.era_score_mat = scipy.zeros([len(self.eras), len(self.eras)])
        self.era_score_reordered_mat = scipy.zeros(
            [len(self.eras), len(self.eras)])
        self.era_i_j_ft_score = dict()

        self.linkage = None
        self.dend_idx = None

        self._compute_eras_proxy()

    def compute_eras_linkage(self):
        # CHOICE
        # linkage method: singlen, ward, centroid ?
        # self.linkage = linkage(self.era_score_mat, 'single')
        # self.linkage = linkage(self.era_score_mat, 'ward')
        self.linkage = linkage(self.era_score_mat, 'centroid')
        self.dend_idx = dendrogram(self.linkage)['leaves']
        self.era_to_dend_idx = [
            self._find_fst_idx_value(self.dend_idx, e_idx - 1)
            for e_idx in self.eras
        ]
        self.era_score_reordered_mat = self.era_score_mat.copy()
        self.era_score_reordered_mat = self.era_score_reordered_mat[:, self.
                                                                    dend_idx]
        self.era_score_reordered_mat = self.era_score_reordered_mat[
            self.dend_idx, :]

    def cluster_weighted_score(self, cl_eras_dend_idx):
        cl_score = 0
        for dend_i in cl_eras_dend_idx:
            for dend_j in cl_eras_dend_idx:
                era_i_j_score = self.era_score_reordered_mat[dend_i, dend_j]
                cl_score += era_i_j_score

        num_era_2 = len(cl_eras_dend_idx) * len(cl_eras_dend_idx)
        cl_score /= num_era_2
        return cl_score

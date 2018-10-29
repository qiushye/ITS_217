"""
created by qiushye on 2018.10.28
python version >= 3
"""

from process.traffic_data import traffic_data
from .Imputation import imputation
from .truncated_svd import truncator
from .halrtc import HaLRTC
import numpy as np
import time
import sys
sys.path.append("..")


class HaLRTC_CSP(imputation):

    def __init__(self, miss_data, K, p, threshold, max_iter=100):

        super(HaLRTC_CSP, self).__init__(miss_data, threshold, max_iter)
        self.K = K
        self.p = p

    def impute(self):
        time_s = time.time()
        sd = self.miss_data.copy()
        td = traffic_data(sd)
        labels = td.cluster_var(self.K)
        Clr_mat = {i: [] for i in range(self.K)}
        for i in range(len(labels)):
            Clr_mat[labels[i]].append(i)

        alpha = [0, 0, 1]
        WT = self.W
        est_data = np.zeros_like(sd)
        for j in range(self.K):
            m_data = sd[labels == j]
            trun_svd = truncator(m_data, self.p)
            trun_svd.truncated_svd()
            lou = 1/trun_svd.SV_list[0]
            halrtc = HaLRTC(m_data, alpha, lou, self.threshold, self.max_iter)
            halrtc.W = WT[labels == j]
            est_data[Clr_mat[j]] = halrtc.impute()
        time_e = time.time()
        self.exec_time = time_e - time_s
        self.est_data = est_data
        return est_data

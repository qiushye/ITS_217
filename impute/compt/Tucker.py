"""
created by qiushye on 2018.10.22
python version >= 3
"""

from Imputation import imputation
from sktensor import tucker, cp
import numpy as np
import time
from sktensor.dtensor import dtensor


class tucker_cpt(imputation):

    def __init__(self, miss_data, W, rank_list, threshold, max_iter=100):
        if len(rank_list) != 3:
            raise RuntimeError('input rank_list error')

        super(tucker_cpt, self).__init__(miss_data, W, threshold, max_iter)
        self.ranks = rank_list

    def impute(self):
        time_s = time.time()
        est_data = self.miss_data.copy()
        SD = dtensor(est_data)
        core1, U1 = tucker.hooi(SD, self.ranks, init='nvecs')
        ttm_data = core1.ttm(U1[0], 0).ttm(U1[1], 1).ttm(U1[2], 2)
        self.est_data = self.W*est_data+(self.W == False)*ttm_data
        time_e = time.time()
        self.exec_time = time_e-time_s

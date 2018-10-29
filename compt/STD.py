"""
created by qiushye on 2018.10.23
python version >= 3
"""

import numpy as np
import time
import sys
from .Imputation import imputation
from sktensor import tucker
from .truncated_svd import truncator
from sktensor.dtensor import dtensor


class STD(imputation):

    def __init__(self, miss_data, threshold, alpha, lam, rank_list, max_iter=500):
        if len(rank_list) != 3:
            raise RuntimeError('input rank_list error')

        super(STD, self).__init__(miss_data, threshold, max_iter)
        self.ranks = rank_list
        self.alpha = alpha
        self.ranks = rank_list
        self.lam = lam

    def restruct(self, core, matrix_list, transpose=False):
        X = core.copy()
        for i in range(len(matrix_list)):
            if transpose:
                X = X.ttm(matrix_list[i].T, i)
            else:
                X = X.ttm(matrix_list[i], i)
        return X

    def impute(self):

        X_ori = self.miss_data.copy()
        core, U_list = tucker.hooi(dtensor(X_ori), self.ranks, init='nvecs')
        X = self.restruct(core, U_list)

        F_diff = sys.maxsize
        iter = 0
        while F_diff > self.threshold and iter < self.max_iter:
            X_pre = X.copy()
            core_pre = core.copy()
            E = self.W * (X_ori - self.restruct(core_pre, U_list))
            for i in range(X.ndim):
                mul1 = (self.W * E).unfold(i)
                if i == 0:
                    mul2 = np.kron(U_list[2], U_list[1])
                elif i == 1:
                    mul2 = np.kron(U_list[2], U_list[0])
                else:
                    mul2 = np.kron(U_list[1], U_list[0])

                mul3 = core_pre.unfold(i).T
                U_list[i] = (1 - self.alpha * self.lam) * U_list[i] + \
                    self.alpha * np.dot(np.dot(mul1, mul2), mul3)

            core_temp = self.restruct(E, U_list, transpose=True)
            core = (1 - self.alpha * self.lam) * \
                core_pre + self.alpha * core_temp
            X = self.restruct(core, U_list)
            F_diff = np.linalg.norm(X-X_pre)
            iter += 1

        return X

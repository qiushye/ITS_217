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

    def __init__(self, miss_data, W, rank_list, alpha, lam, threshold, max_iter=500):
        if len(rank_list) != 3:
            raise RuntimeError('input rank_list error')

        super(STD, self).__init__(miss_data, W, threshold, max_iter)
        self.ranks = rank_list
        self.alpha = alpha
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
        time_s = time.time()
        X_ori = self.miss_data.copy()
        core, U_list = tucker.hooi(dtensor(X_ori), self.ranks, init='nvecs')
        X = self.restruct(core, U_list)

        F_diff = sys.maxsize
        iter = 0
        while iter < self.max_iter:
            F_diff_pre = F_diff
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
            F_diff = np.linalg.norm(X - X_pre)
            if abs(F_diff-F_diff_pre) > self.threshold:
                break
            print('STD:', F_diff)
            iter += 1
        time_e = time.time()
        self.exec_time = time_e - time_s
        self.est_data = X
        return X

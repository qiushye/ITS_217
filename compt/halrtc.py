"""
created by qiushye on 2018.10.23
python version >= 3
"""

import time
import numpy as np
from .Imputation import imputation
from sktensor.dtensor import dtensor


class HaLRTC(imputation):

    def __init__(self, miss_data, alpha, lou, threshold, max_iter=100):
        if len(alpha) != 3:
            raise RuntimeError('input rank_list error')

        super(HaLRTC, self).__init__(miss_data, threshold, max_iter)
        self.alpha_vec = alpha
        self.lou = lou

    def impute(self):
        time_s = time.time()
        alpha = self.alpha_vec
        lou = self.lou
        X = self.miss_data.copy()
        Y, M = {}, {}
        N = len(X.shape)
        W1 = (self.W == False)
        T_temp = X.copy()

        for _ in range(N):
            M[_] = dtensor(np.zeros(np.shape(X)))
            Y[_] = dtensor(np.zeros(np.shape(X)))

        for _ in range(self.max_iter):
            X_pre = X.copy()
            for i in range(N):
                SD = dtensor(X_pre)
                Matrix = SD.unfold(i)+1/lou*(Y[i].unfold(i))

                U, sigma, VT = np.linalg.svd(Matrix, 0)
                row_s = len(sigma)
                mat_sig = np.zeros((row_s, row_s))
                for ii in range(row_s):
                    mat_sig[ii, ii] = max(sigma[ii]-alpha[i]/lou, 0)
                M[i] = (np.dot(np.dot(U, mat_sig), VT[:row_s, :])).fold()

            T_temp = (np.sum([M[j]-1/lou*Y[j] for j in range(N)], axis=0))/N
            X[W1] = T_temp[W1]
            X_Fnorm = np.sum((X-X_pre)**2)
            if X_Fnorm < self.threshold:
                break
            for i in range(N):
                Y[i] -= lou * (M[i] - X)

        time_e = time.time()
        self.exec_time = time_e - time_s
        self.est_data = X
        return X

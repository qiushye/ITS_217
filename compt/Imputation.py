"""
created by qiushye on 2018.10.22
python version >= 3
"""
import numpy as np


class imputation:

    def __init__(self, miss_data, threshold, max_iter=100):
        # miss_data type should be numpy.ndarray
        if type(miss_data) != np.ndarray:
            raise RuntimeError('input type error')

        self.miss_data = miss_data
        self.W = miss_data > 0
        self.shape = miss_data.shape
        self.miss_ratio = self.W.sum()/miss_data.size
        self.threshold = threshold
        self.max_iter = max_iter
        self.est_data = miss_data.copy()
        self.exec_time = 0

        if self.miss_ratio == 0:
            raise RuntimeError('input data is complete')

    def pre_impute(self, day_axis=1):
        sparse_data = self.miss_data
        pos = np.where(self.W == False)
        for p in range(len(pos[0])):
            i, j, k = pos[0][p], pos[1][p], pos[2][p]
            if day_axis == 0:
                if (sparse_data[:, j, k] > 0).sum() > 0:
                    sparse_data[i, j, k] = np.sum(
                        sparse_data[:, j, k])/(sparse_data[:, j, k] > 0).sum()
                else:
                    sparse_data[i, j, k] = np.sum(
                        sparse_data[i, :, :])/(sparse_data[i, :, :] > 0).sum()
            elif day_axis == 1:
                if (sparse_data[i, :, k] > 0).sum() > 0:
                    sparse_data[i, j, k] = np.sum(
                        sparse_data[i, :, k])/(sparse_data[i, :, k] > 0).sum()
                else:
                    sparse_data[i, j, k] = np.sum(
                        sparse_data[:, j, :])/(sparse_data[:, j, :] > 0).sum()
            else:
                if (sparse_data[i, j, :] > 0).sum() > 0:
                    sparse_data[i, j, k] = np.sum(
                        sparse_data[i, j, :])/(sparse_data[i, :, k] > 0).sum()
                else:
                    sparse_data[i, j, k] = np.sum(
                        sparse_data[:, j, :])/(sparse_data[:, j, :] > 0).sum()

        self.miss_data = sparse_data
        return sparse_data

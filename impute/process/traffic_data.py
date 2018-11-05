"""
created by qiushye on 2018.10.26
python version >= 3
"""

import numpy as np
import os
import scipy.io as scio
from sklearn.cluster import KMeans


class traffic_data:

    def __init__(self, ori_data):
        if type(ori_data) != np.ndarray:
            raise RuntimeError('input type error')

        self.shape = ori_data.shape
        self.ori_W = (ori_data > 0)
        self.ori_miss_rate = self.ori_W.sum()/ori_data.size
        if self.ori_miss_rate > 0:
            W_miss = np.where(ori_data <= 0)
            for i in range(len(W_miss[0])):
                pos1, pos2, pos3 = W_miss[0][i], W_miss[1][i], W_miss[2][i]
                neigh_info = []
                for n2 in (1, -1):
                    for n3 in (1, -1):
                        try:
                            temp = ori_data[pos1, pos2+n2, pos3+n3]
                            neigh_info.append(temp)
                        except:
                            pass
                if sum(neigh_info) > 0:
                    ori_data[pos1, pos2, pos3] = sum(
                        neigh_info)/(np.array(neigh_info) > 0).sum()

        self.ori_data = ori_data

    def generate_miss(self, miss_ratio, miss_type="rand", miss_path=""):

        if miss_path != '' and os.path.exists(miss_path):
            miss_data = scio.loadmat(miss_path)['Speed']

            true_miss_ratio = (miss_data > 0).sum() / miss_data.size
            return miss_data, true_miss_ratio

        miss_data = self.ori_data.copy()
        dshape = self.shape
        if miss_type == "rand":

            rand_ts = np.random.random_sample(dshape)
            zero_ts = np.zeros(dshape)
            miss_data = miss_data*(rand_ts > miss_ratio) + \
                zero_ts*(rand_ts <= miss_ratio)

        else:
            rand_ts = np.random.random_sample(dshape[:-1])
            S = np.rint(rand_ts+0.5-miss_ratio)
            W_cont = np.zeros(dshape)
            for k in range(dshape[2]):
                W_cont[:, :, k] = S[:, :]
            miss_data = miss_data*W_cont

        true_miss_ratio = (miss_data > 0).sum() / miss_data.size

        if len(miss_path) > 3 and miss_path[-3:] == 'mat':
            scio.savemat(miss_path, {'Speed': miss_data})

        return miss_data, round(true_miss_ratio, 1)

    def cluster_var(self, K):
        data = self.ori_data
        ds = self.shape
        var_mat = np.zeros((ds[0], ds[1]))
        for r in range(ds[0]):
            for d in range(ds[1]):
                var_mat[r, d] = np.var(data[r, d, :])

        clf = KMeans(n_clusters=K)
        S = clf.fit(var_mat)
        return S.labels_

    def evaluate(self, est_data, W, precision=4):
        if est_data.shape != self.shape:
            raise RuntimeError(
                'the shape of est_data is not same as that of ori_data')
        W_miss = (W == False)
        ori_data = self.ori_data
        diff_data = np.zeros_like(est_data)+W_miss*(est_data-ori_data)
        rmse = float((np.sum(diff_data ** 2) / W_miss.sum()) ** 0.5)
        mre_mat = np.zeros_like(est_data)
        mre_mat[W_miss] = np.abs(
            (est_data[W_miss]-ori_data[W_miss])/ori_data[W_miss])
        mape = float(np.sum(mre_mat)/W_miss.sum())*100
        # rse = float(np.sum(diff_data**2)**0.5/np.sum(ori_data[W_miss]**2)**0.5)
        mae = float(np.sum(np.abs(diff_data))/W_miss.sum())
        return round(rmse, precision), round(mape, max(precision-2, 0)), round(mae, precision)

"""
created by qiushye on 2018.10.26
python version >= 3
"""

import numpy as np
import scipy.io as scio


class traffic_data:

    def __init__(self, ori_data):
        if type(ori_data) != np.ndarray:
            raise RuntimeError('input type error')

        self.shape = ori_data.shape
        self.ori_miss_rate = (ori_data > 0).sum()/ori_data.size
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
            scio.savemat(miss_path, {'Speed': miss_path})

        return miss_data, true_miss_ratio

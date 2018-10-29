"""
created by qiushye on 2018.10.28
python version >= 3
"""

import scipy.io as scio
from process.traffic_data import traffic_data
from compt.Imputation import imputation
from compt.bpca_cpt import BPCA_CPT
from compt.halrtc import HaLRTC
from compt.halrtc_csp import HaLRTC_CSP
from compt.STD import STD
from compt.truncated_svd import truncator
import pandas as pd
import numpy as np
import os

ori_path = 'D:/GZ_data/60days_tensor.mat'
data_dir = 'D:/ITS_217/data/'
img_dir = 'D:/ITS_217/img_test/'

threshold = 1e-4
max_iter = 100
RAND_MISS = "rand"
CONT_MISS = "cont"


def init():
    if not os.path.exists(ori_path):
        raise RuntimeError("ori_data path doesn't exist")

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)


def compare_methods(ori_data, miss_type, number=8, p=0.7):
    td = traffic_data(ori_data)

    methods = ['HAI', 'BPCA', 'STD', 'HaLRTC', 'HaLRTC-CSP']
    eva_list = ['RMSE', 'MAE', 'MRE', 'RSE', 'Run_Time']
    result = {}
    miss_list = []

    for i in range(number):
        miss_ratio = round(0.1 * (i + 1), 2)
        miss_path = data_dir+'miss_' + \
            str(miss_ratio)+''.join(['_'+str(ch) for ch in td.shape])+'.mat'
        miss_data, tm_ratio = td.generate_miss(
            miss_ratio, miss_type, miss_path)
        # print(miss_data)
        miss_list.append(tm_ratio)
        W = miss_data > 0
        method_index = 0

        def save_result(key, method, n):
            df = pd.DataFrame([0, 0, 0, 0, 0], columns=[-1], index=eva_list)
            if key not in result:
                result[key] = df
            est_data = method.impute()
            rW = W | (td.ori_W == False)
            eva = list(td.evaluate(est_data, rW))
            eva.append(method.exec_time)
            # print(td.evaluate(est_data, rW))
            result[key][n] = eva

        # history average imputation(HAI)
        hai = imputation(miss_data, threshold, max_iter)
        save_result(methods[method_index], hai, i)
        method_index += 1
        miss_data = hai.impute()  # 其他填充以预填充后的数据为基础
        # eva = list(td.evaluate(miss_data, W)).append(hai.exec_time)
        # df = pd.DataFrame(eva, columns=['HAI'])

        trun_svd = truncator(miss_data, p)
        trun_svd.truncated_svd()

        # BPCA

        mult_components = trun_svd.rank_list[0]
        bpca = BPCA_CPT(miss_data, mult_components, threshold, max_iter)
        save_result(methods[method_index], bpca, i)
        method_index += 1

        # STD
        ap, lm = 2e-10, 0.01
        std = STD(miss_data, threshold, ap, lm, trun_svd.rank_list, max_iter)
        save_result(methods[method_index], std, i)
        method_index += 1

        # halrtc
        lou = 1 / trun_svd.SV_list[0]
        alpha = [1/3, 1/3, 1/3]
        halrtc = HaLRTC(miss_data, alpha, lou, threshold, max_iter)
        save_result(methods[method_index], halrtc, i)
        method_index += 1

        # halrtc-csp
        K = 4
        halrtc_csp = HaLRTC_CSP(miss_data, K, p, threshold, max_iter)
        save_result(methods[method_index], halrtc_csp, i)
        method_index += 1

        print(result)
        break


if __name__ == "__main__":

    init()
    ori_data = scio.loadmat(ori_path)['tensor']
    compare_methods(ori_data, RAND_MISS)

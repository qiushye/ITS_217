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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

ori_path = 'D:/GZ_data/60days_tensor.mat'
data_dir = 'D:/ITS_217/data/'
result_dir = 'D:/ITS_217/result/'

threshold = 1e-4
max_iter = 100
RAND_MISS = "rand"
CONT_MISS = "cont"


def init():
    if not os.path.exists(ori_path):
        raise RuntimeError("ori_data path doesn't exist")

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


def compare_methods(ori_data, miss_type, number=8, p=0.7):
    td = traffic_data(ori_data)

    methods = ['HAI', 'BPCA', 'STD', 'HaLRTC', 'HaLRTC-CSP']
    eva_list = ['RMSE', 'MRE', 'MAE', 'Run_Time']
    result = {}
    miss_list = []

    for i in range(number):
        miss_ratio = round(0.1 * (i + 1), 2)
        miss_path = data_dir+'miss_' + miss_type + '_' + \
            str(miss_ratio)+''.join(['_'+str(ch) for ch in td.shape])+'.mat'
        miss_data, tm_ratio = td.generate_miss(
            miss_ratio, miss_type, miss_path)
        # print(miss_data)
        miss_list.append(tm_ratio)
        W = miss_data > 0
        method_index = 0

        def save_result(key, method, n):
            est_data = method.impute()
            rW = W | (td.ori_W == False)
            eva = list(td.evaluate(est_data, rW))
            print(key, eva)
            eva.append(method.exec_time)
            if key not in result:
                df = pd.DataFrame(eva, columns=[n], index=eva_list)
                result[key] = df
            else:
                result[key][n] = eva

        # history average imputation(HAI)
        hai = imputation(miss_data, W, threshold, max_iter)
        save_result(methods[method_index], hai, i)
        method_index += 1
        miss_data = hai.impute()  # 其他填充以预填充后的数据为基础

        tensor_trun_svd = truncator(miss_data, p)
        tensor_trun_svd.truncated_svd()

        # BPCA
        multi_components = []
        for i in range(miss_data.shape[0]):
            matrix_trun_svd = truncator(miss_data[i], p)
            matrix_trun_svd.truncated_svd()
            multi_components.append(matrix_trun_svd.rank_list[0])
        bpca = BPCA_CPT(miss_data, W, multi_components, threshold, max_iter)
        save_result(methods[method_index], bpca, i)
        method_index += 1

        # STD
        ap, lm = 2e-10, 0.01
        std = STD(miss_data, W, tensor_trun_svd.rank_list,
                  ap, lm, threshold, max_iter)
        save_result(methods[method_index], std, i)
        method_index += 1

        # halrtc
        lou = 1 / tensor_trun_svd.SV_list[0]
        alpha = [1/3, 1/3, 1/3]
        halrtc = HaLRTC(miss_data, W, alpha, lou, threshold, max_iter)
        save_result(methods[method_index], halrtc, i)
        method_index += 1

        # halrtc-csp
        K = 4
        halrtc_csp = HaLRTC_CSP(miss_data, W, K, p, threshold, max_iter)
        save_result(methods[method_index], halrtc_csp, i)
        method_index += 1

        print(result['HAI'].loc['RMSE'])
        break

    text_path = result_dir + 'compare_methods.txt'
    fw = open(text_path, 'w')
    fw.write('methods:'+','.join(list(methods))+'\n')
    fw.write('Missing Rate (%):' + ','.join(list(map(str, miss_list))) + '\n')

    def obtain_ticks(min_value, max_value, interval):
        start = min_value
        ticks = []
        while start <= max_value:
            ticks.append(start)
            start += interval

        return ticks

    metric_dict = {'RMSE': 'km/h', 'MAE': 'km/h', 'MRE': '%', 'Run_Time': 's'}
    MK = ['o', 'o', '*', '*', 'x', 'x']     # marks
    CR = ['r', 'b', 'y', 'r', 'b', 'y']     # colors
    Xlim = [0, 90]
    Yminor = {'RMSE': 0.5, 'MAE': 0.2, 'MRE': 1, 'Run_Time': 50}
    Ylims = {'RMSE': [2.4, 5.6], 'MAE': [1.6, 3.6],
             'MRE': [5, 13], 'Run_Time': [0, 500]}
    Yticks = {}
    for eva in Ylims:
        Yticks[eva] = obtain_ticks(Ylims[eva][0], Ylims[eva][1], Yminor[eva])

    for eva in eva_list:
        img_path = result_dir+'compare_methods_'+miss_type+'_'+eva+'.png'

        ax = plt.subplot()
        ax.set_xlabel('Missing Rate (%)')
        ax.set_ylabel(eva + ' (' + metric_dict[eva] + ')')
        ax.set_xlim(Xlim)
        ax.set_ylim(Ylims[eva])

        ax.set_yticks(Yticks[eva])

        fw.write(eva + ':\n')
        nl = 0
        for method in methods:
            ax.plot(miss_list, result[method].loc[
                eva], color=CR[nl], marker=MK[nl], label='$'+method+'$')
            fw.write(','.join(list(map(str, result[method].loc[eva]))) + '\n')
            nl += 1
        plt.legend(loc='best')
        plt.savefig(img_path)
        plt.close()
    fw.close()
    return


def compare_C(ori_data, miss_ratio, miss_type, p=0.7):
    td = traffic_data(ori_data)
    miss_path = data_dir+'miss_' + miss_type + '_' + \
        str(miss_ratio)+''.join(['_'+str(ch) for ch in td.shape])+'.mat'
    miss_data, tm_ratio = td.generate_miss(
        miss_ratio, miss_type, miss_path)
    W = miss_data > 0

    hai = imputation(miss_data, W, threshold, max_iter)
    miss_data = hai.impute()

    C_list = list(range(2, 11))
    eva_list = ['RMSE', 'MRE', 'MAE', 'Run_Time']
    df = pd.DataFrame(0, columns=[0], index=eva_list)
    for C in C_list:
        halrtc_csp = HaLRTC_CSP(miss_data, W, C, p, threshold, max_iter)
        est_data = halrtc_csp.impute()
        rW = W | (td.ori_W == False)
        eva = list(td.evaluate(est_data, rW))
        eva.append(halrtc_csp.exec_time)
        df[C] = eva


if __name__ == "__main__":

    init()
    ori_data = scio.loadmat(ori_path)['tensor']
    compare_methods(ori_data, RAND_MISS)

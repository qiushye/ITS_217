"""
created by qiushye on 2018.10.28
python version >= 3
"""

import scipy.io as scio
from compt.traffic_data import traffic_data
from compt.imputation import imputation
from compt.bpca_cpt import BPCA_CPT
from compt.halrtc import HaLRTC
from compt.halrtc_csp import HaLRTC_CSP
from compt.STD import STD
from compt.truncated_svd import truncator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns

ori_path = 'D:/GZ_data/60days_tensor.mat'
cur_dir = os.path.split(os.path.realpath(__file__))[0]
data_dir = cur_dir + '/data/'
result_dir = cur_dir + '/result/'

threshold = 1e-4
max_iter = 100
RAND_MISS = "rand"
CONT_MISS = "cont"
metric_dict = {'RMSE': 'km/h', 'MAE': 'km/h', 'MRE': '%', 'Run_Time': 's'}


def init():
    if not os.path.exists(ori_path):
        raise RuntimeError("ori_data path doesn't exist")

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


def obtain_ticks(min_value, max_value, interval):
    start = min_value
    ticks = []
    while start <= max_value:
        ticks.append(start)
        start += interval

    return ticks


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
        miss_data, tm_ratio = td.generate_miss(miss_ratio, miss_type,
                                               miss_path)
        miss_list.append(tm_ratio * 100)
        W = miss_data > 0
        method_index = 0
        print(miss_ratio, tm_ratio)

        def save_result(key, method, n):
            est_data = method.impute()
            rW = W | (td.ori_W == False)
            eva = list(td.evaluate(est_data, rW))
            print(key, eva)
            eva.append(round(method.exec_time, 1))
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
        for j in range(miss_data.shape[0]):
            matrix_trun_svd = truncator(miss_data[j], p)
            matrix_trun_svd.truncated_svd()
            multi_components.append(matrix_trun_svd.rank_list[0])
        bpca = BPCA_CPT(miss_data, W, multi_components, threshold, max_iter)
        save_result(methods[method_index], bpca, i)
        method_index += 1

        # STD
        ap, lm = 2e-10, 0.01
        std = STD(miss_data, W, tensor_trun_svd.rank_list, ap, lm, threshold,
                  max_iter)
        save_result(methods[method_index], std, i)
        method_index += 1

        # halrtc
        lou = 1 / tensor_trun_svd.SV_list[0]
        alpha = [1 / 3, 1 / 3, 1 / 3]
        halrtc = HaLRTC(miss_data, W, alpha, lou, threshold, max_iter)
        save_result(methods[method_index], halrtc, i)
        method_index += 1

        # halrtc-csp
        K = 4
        halrtc_csp = HaLRTC_CSP(miss_data, W, K, p, threshold, max_iter)
        save_result(methods[method_index], halrtc_csp, i)
        method_index += 1

        print(result['HAI'].loc['RMSE'])
        # break

    text_path = result_dir + 'compare_methods.txt'
    fw = open(text_path, 'w')
    fw.write('methods:' + ','.join(list(methods)) + '\n')
    fw.write('Missing Rate (%):' + ','.join(list(map(str, miss_list))) + '\n')

    MK = ['o', 'o', '*', '*', 'x', 'x']  # marks
    CR = ['r', 'b', 'y', 'r', 'b', 'y']  # colors
    Xlim = [0, 90]
    Yminor = {'RMSE': 0.5, 'MAE': 0.2, 'MRE': 1, 'Run_Time': 50}
    Ylims = {
        'RMSE': [2.4, 5.6],
        'MAE': [1.6, 3.6],
        'MRE': [5, 13],
        'Run_Time': [0, 500]
    }
    Yticks = {}
    for eva in Ylims:
        Yticks[eva] = obtain_ticks(Ylims[eva][0], Ylims[eva][1], Yminor[eva])

    for eva in eva_list:
        img_path = result_dir + 'compare_methods_' + miss_type + '_' + eva + '.pdf'

        ax = plt.subplot()
        ax.set_xlabel('Missing Rate (%)')
        ax.set_ylabel(eva + ' (' + metric_dict[eva] + ')')
        ax.set_xlim(Xlim)
        ax.set_ylim(Ylims[eva])

        ax.set_yticks(Yticks[eva])
        ax.yaxis.grid(True, linestyle='--')
        ax.xaxis.grid(True, linestyle='--')

        fw.write(eva + ':\n')
        nl = 0
        for method in methods:
            ax.plot(
                miss_list,
                result[method].loc[eva].tolist(),
                color=CR[nl],
                marker=MK[nl],
                label='$' + method + '$')
            fw.write(','.join(list(map(str, result[method].loc[eva]))) + '\n')
            nl += 1
        plt.legend(loc='best')
        plt.savefig(img_path, format='pdf')
        plt.close()
    fw.close()
    print(miss_list)
    return


def compare_C(ori_data, miss_ratio, miss_type, p=0.7):
    td = traffic_data(ori_data)
    miss_path = data_dir+'miss_' + miss_type + '_' + \
        str(miss_ratio)+''.join(['_'+str(ch) for ch in td.shape])+'.mat'
    miss_data, tm_ratio = td.generate_miss(miss_ratio, miss_type, miss_path)
    W = miss_data > 0

    hai = imputation(miss_data, W, threshold, max_iter)
    miss_data = hai.impute()

    C_list = list(range(2, 31))
    eva_list = ['RMSE', 'MRE', 'MAE', 'Run_Time']
    df = pd.DataFrame(0, columns=[0], index=eva_list)
    for C in C_list:
        halrtc_csp = HaLRTC_CSP(miss_data, W, C, p, threshold, max_iter)
        est_data = halrtc_csp.impute()
        rW = W | (td.ori_W == False)
        eva = list(td.evaluate(est_data, rW))
        eva.append(halrtc_csp.exec_time)
        df[C] = eva

    text_path = result_dir + 'compare_C.txt'
    Xlabel = 'Clustering Number C'
    fw = open(text_path, 'w')
    fw.write(Xlabel + ':' + ','.join(list(map(str, C_list))) + 'for ' +
             str(tm_ratio) + ' missing\n')

    # Xminor = 2
    Yminor = {'RMSE': 0.05, 'MAE': 0.05, 'MRE': 0.1, 'Run_Time': 10}
    Xlim = [0, 32]
    Ylims = {
        'RMSE': [2.6, 3.0],
        'MAE': [1.8, 2.1],
        'MRE': [6.0, 6.6],
        'Run_Time': [30, 100]
    }
    Yticks = {}
    for eva in Ylims:
        Yticks[eva] = obtain_ticks(Ylims[eva][0], Ylims[eva][1], Yminor[eva])

    for eva in eva_list:
        img_path = result_dir + 'compare_C_' + miss_type + '_' + eva + '.pdf'
        ax = plt.subplot()
        ax.set_xlabel(Xlabel)
        ax.set_ylabel(eva + ' (' + metric_dict[eva] + ')')
        ax.set_xlim(Xlim)
        ax.set_ylim(Ylims[eva])

        ax.plot(C_list, df.loc[eva].tolist(), color='r', marker='o')
        ax.legend(loc='best')
        plt.savefig(img_path, format='pdf')
        plt.close()

        fw.write(eva + ':\n')
        fw.write(','.join(list(map(str, df[eva]))) + '\n')

    fw.close()
    return


def ori_imputation(ori_data, miss_ratio, miss_type):
    td = traffic_data(ori_data)
    miss_path = data_dir+'miss_' + miss_type + '_' + \
        str(miss_ratio)+''.join(['_'+str(ch) for ch in td.shape])+'.mat'
    miss_data, _ = td.generate_miss(miss_ratio, miss_type, miss_path)

    W = miss_data > 0
    ds = miss_data.shape
    rW = W | (td.ori_W == False)

    C = 4
    p = 0.7
    halrtc_csp = HaLRTC_CSP(miss_data, W, C, p, threshold, max_iter)
    est_data = halrtc_csp.impute()
    plt.xlabel('estimated speed (km/h)')
    plt.ylabel('original speed (km/h)')
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.plot([0, 50], [0, 50], color='b', linewidth=2)
    #plt.scatter(1, 2, s=10, c='r')
    r = 1
    for j in range(ds[1]):
        for k in range(ds[2]):
            if not rW[r, j, k]:
                plt.scatter(
                    td.ori_data[r, j, k], est_data[r, j, k], s=10, c='r')
    plt.savefig(result_dir + 'est_ori.pdf')
    plt.close()
    return


def heatmap(ori_data, miss_ratio, miss_type):
    td = traffic_data(ori_data)
    miss_path = data_dir+'miss_' + miss_type + '_' + \
        str(miss_ratio)+''.join(['_'+str(ch) for ch in td.shape])+'.mat'
    miss_data, _ = td.generate_miss(miss_ratio, miss_type, miss_path)
    shape = td.shape

    ax = plt.subplot()
    sns.heatmap(
        miss_data[0],
        cmap='RdBu',
        linewidths=0.05,
        xticklabels=8,
        yticklabels=5)
    ax.invert_yaxis()
    ax.set_xlim([0, shape[2]])
    ax.set_ylim([0, shape[1]])
    ax.set_xlabel('time periods')
    ax.set_ylabel('days')
    plt.savefig(result_dir + 'heatmap.pdf', format='pdf')
    plt.close()
    return


# if __name__ == "__main__":

#     init()
#     miss_ratio = 0.2
#     ori_data = scio.loadmat(ori_path)['tensor']
#     # compare_methods(ori_data, RAND_MISS, number=8)
#     # compare_C(ori_data, miss_ratio, RAND_MISS, p=0.7)
#     # ori_imputation(ori_data, miss_ratio, RAND_MISS)
#     heatmap(ori_data, miss_ratio, RAND_MISS)

# -*- coding: UTF-8 -*-
import math
import os
import numpy as np
import copy
import time
import random
import pandas as pd
import road_network
"""
created by qiushye on 2018.11.12
python version >= 3
"""
import sys
sys.path.append('.')
from impute.compt.halrtc_csp import HaLRTC_CSP
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from population import population

from init import dates, cur_dir, data_dir, result_dir, weekday, weekend, weekday_index, weekend_index
dates = weekday


def init():  # 初始化目录

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


def speed_assign(file_path):  # 速度等级分布统计
    df = pd.read_excel(file_path, skip_footer=1)
    if 'speed' not in df.columns:
        df.columns = ['road', 'vol', 'speed', 'last-update-time']
    df.drop_duplicates('last-update-time', 'first', inplace=True)
    df.index = list(map(str, df['last-update-time']))

    # road = file_path.split('.')[0].split('Viewer')[1]
    v_array = np.zeros((11, len(dates)))
    for t in df.index:
        if df['speed'][t] == '':
            continue

        v = float(df['speed'][t])
        date = t[:10]
        if '/' in date:
            date.replace('/', '-')

        if date not in dates:
            continue
        row = dates.index(date)
        if v >= 100:
            col = 10
        else:
            col = int(v / 10)
        v_array[col, row] += 1

    return v_array


def road_analyse(raw_dir, all_roads, roads_path):  # 分析路段整体速度按日期的变化情况
    # all_roads = []
    # with open(roads_path, 'r') as f:
    #     for line in f:
    #         row = line.strip().split(',')
    #         all_roads.append(row[0])

    v_array = np.zeros((11, len(dates)))
    for file in os.listdir(raw_dir):
        if file[-3:] != 'xls':
            continue
        id = file.split('.')[0].split('Viewer')[1]
        if id not in all_roads:
            continue
        v_array += speed_assign(raw_dir + file)

        ax = plt.subplot()
        sns.heatmap(v_array, cmap='RdBu', linewidths=0.05)
        ax.invert_yaxis()
        # ax.set_xlim([0, 14])
        # ax.set_ylim([0, 10])
        ax.set_xlabel('dates')
        ax.set_ylabel('speed level(degree=10km/h)')
        plt.savefig(result_dir + id + '_dates_speed.png', bbox_inches='tight')
        plt.close()
    return


def var_assign(unseed_path, interval):  # 非种子路段方差变化情况
    uns_roads = []
    with open(unseed_path, 'r') as f:
        uns_roads = f.readline().strip().split(',')
    files_dir = data_dir + str(interval) + '_impute/'
    road_data = []
    for r in uns_roads:
        file = r + '.csv'
        df = pd.read_csv(files_dir + file, index_col=0)
        arr = df.values
        road_data.append(arr)

    speed_tensor = np.array(road_data)
    var_matrix = np.zeros((speed_tensor.shape[2], len(uns_roads)))
    sp = var_matrix.shape
    print('shape:', sp)
    fw = open(result_dir + 'var_assign.csv', 'w')
    fw.write(',' + ','.join(uns_roads) + '\n')
    for i in range(sp[0]):
        fw.write(str(i))
        for j in range(sp[1]):
            var_matrix[i, j] = round(np.var(speed_tensor[j, weekday_index, i]))
            fw.write(',' + str(var_matrix[i, j]))
        fw.write('\n')

    fw.close()
    ax = plt.subplot()
    sns.heatmap(var_matrix, cmap='RdBu', linewidths=0.05)
    ax.set_ylabel('time periods')
    ax.set_xlabel('roads')
    # ax.yaxis.grid(True, linestyle='--')
    # ax.xaxis.grid(True, linestyle='--')
    plt.savefig(result_dir + 'var_assign.png', bbox_inches='tight')
    plt.close()

    return


def speed_extract(file_path, interval, data_dir):  # 处理源采样数据，聚合成设定时间间隔的速度数据
    df = pd.read_excel(file_path, skip_footer=1)
    if 'speed' not in df.columns:
        df.columns = ['road', 'vol', 'speed', 'last-update-time']
    df.drop_duplicates('last-update-time', 'first', inplace=True)
    df.index = list(map(str, df['last-update-time']))

    periods = 24 * 60 // interval
    speed_dict = dict()
    # v_list = {}
    for t in df.index:
        if df['speed'][t] == '':
            continue

        v = float(df['speed'][t])

        date = t[:10]
        if '/' in date:
            date.replace('/', '-')

        time_series = list(map(int, t[11:].split(':')))
        times = time_series[0] / (interval / 60) + round(
            time_series[1] / interval)
        time_period = times % periods
        date_plus = times // periods
        if date_plus:
            new_day = str(int(date[8:]) + 1)
            date = t[:8] + (2 - len(new_day)) * '0' + new_day

        if date not in dates:
            continue
        if v < 10 or v > 100:
            continue

        if date not in speed_dict:
            speed_dict[date] = {}

        if time_period not in speed_dict[date]:
            speed_dict[date][time_period] = [0, 0]

        speed_dict[date][time_period][0] += v
        speed_dict[date][time_period][1] += 1

    speed_arr = np.zeros((len(dates), periods))

    for i in range(len(dates)):
        if dates[i] not in speed_dict:
            continue
        for j in range(periods):
            if j not in speed_dict[dates[i]]:
                continue
            [v_sum, count] = speed_dict[dates[i]][j]
            if v_sum > 0:
                speed_arr[i][j] = round(v_sum / count, 1)

    return speed_arr


def complete(roads_path, raw_dir, interval,
             data_dir):  # 对所有路段的数据源进行预处理，并用HaLRTC-CSP做预填充
    roads = []
    road_data = []
    periods = 24 * 60 // interval
    with open(roads_path, 'r') as f:
        for line in f:
            row = line.strip().split(',')
            roads.append(row[0])

    order_roads = []  # 文件打开顺序
    for file in os.listdir(raw_dir):
        if file[-3:] != 'xls':
            continue
        id = file.split('.')[0].split('Viewer')[1]
        if id not in roads:
            continue
        arr = speed_extract(raw_dir + file, interval, data_dir)
        # continue
        road_data.append(arr)
        order_roads.append(id)

        before_impute_path = data_dir + str(interval) + '_before_impute/'
        if not os.path.exists(before_impute_path):
            os.mkdir(before_impute_path)
        fw = open(before_impute_path + id + '.csv', 'w')
        fw.write(',' + ','.join([str(i) for i in range(periods)]) + '\n')

        for j in range(len(dates)):
            fw.write(dates[j] + ',')
            fw.write(','.join([str(speed) for speed in arr[j]]) + '\n')
        fw.close()

    K = 4
    p = 0.7
    threshold = 1e-4
    max_iter = 100

    road_tensor = np.array(road_data)
    W = road_tensor > 0
    halrtc_csp = HaLRTC_CSP(road_tensor, W, K, p, threshold, max_iter)
    new_data = halrtc_csp.impute()

    for i in range(len(order_roads)):
        impute_path = data_dir + str(interval) + '_impute/'
        if not os.path.exists(impute_path):
            os.mkdir(impute_path)
        fw = open(impute_path + order_roads[i] + '.csv', 'w')
        fw.write(',' + ','.join([str(i) for i in range(periods)]) + '\n')
        speed_arr = new_data[i]
        for j in range(len(dates)):
            fw.write(dates[j] + ',')
            fw.write(','.join([str(speed) for speed in speed_arr[j]]) + '\n')
        fw.close()
    return W


def mode_miss(data_dir, interval, time_period):  # 各个维度的缺失情况
    # periods = 24 * 60 // interval
    road_data = []
    roads = []
    miss_dir = data_dir + str(interval) + '_before_impute/'
    for file in os.listdir(miss_dir):
        file_path = miss_dir + file
        id = file.split('.')[0]
        df = pd.read_csv(file_path, index_col=0)

        arr = df.values
        roads.append(id)
        road_data.append(arr)

    speed_tensor = np.array(road_data)
    '''
    miss_rate = []
    for i in range(len(roads)):
        M = speed_tensor[i]
        miss_rate.append((M <= 0).sum() / M.size)
    plt.scatter(list(range(len(roads))), miss_rate)
    plt.xlabel("roads")
    plt.ylabel("missing_rate")
    plt.savefig(result_dir + 'roads_missing.png', bbox_inches='tight')
    plt.close()

    miss_rate = []
    for i in range(len(dates)):
        M = speed_tensor[:, i, :]
        miss_rate.append((M <= 0).sum() / M.size)
    plt.scatter(list(range(len(dates))), miss_rate)
    plt.xlabel("dates")
    plt.ylabel("missing_rate")
    plt.savefig(result_dir + 'dates_missing.png', bbox_inches='tight')
    plt.close()
    # plt.show()

    miss_rate = []
    for i in range(periods):
        M = speed_tensor[:, :, i]
        miss_rate.append((M <= 0).sum() / M.size)
    plt.scatter(list(range(periods)), miss_rate)
    plt.xlabel("periods")
    plt.ylabel("missing_rate")
    plt.savefig(result_dir + 'periods_missing.png', bbox_inches='tight')
    plt.close()
    '''
    '''
    weekday_mean_speed = []
    weekend_mean_speed = []
    weekday_var, weekend_var = [], []
    for i in range(len(roads)):
        weekday_M = speed_tensor[i, weekday_index, int(time_period)]
        weekend_M = speed_tensor[i, weekend_index, int(time_period)]
        weekday_mean_speed.append(weekday_M.sum() / (weekday_M > 0).sum())
        weekend_mean_speed.append(weekend_M.sum() / (weekend_M > 0).sum())
        weekday_var.append(np.var(weekday_M))
        weekend_var.append(np.var(weekend_M))
    plt.scatter(
        list(range(len(roads))),
        weekday_mean_speed,
        marker='o',
        label='weekday')
    plt.scatter(
        list(range(len(roads))),
        weekend_mean_speed,
        marker='*',
        label='weekend')
    plt.xlabel("roads")
    plt.ylabel("mean_speed")
    plt.legend(loc='best')
    plt.savefig(result_dir + 'roads_mean_speed.png', bbox_inches='tight')
    plt.close()

    plt.scatter(
        list(range(len(roads))), weekday_var, marker='o', label='weekday')
    plt.scatter(
        list(range(len(roads))), weekend_var, marker='*', label='weekend')
    plt.xlabel("roads")
    plt.ylabel("variance")
    plt.legend(loc='best')
    plt.savefig(result_dir + 'roads_var.png', bbox_inches='tight')
    plt.close()
    '''
    weekday1_roads_speed = speed_tensor[:, weekend_index[0], int(time_period)]
    weekday2_roads_speed = speed_tensor[:, weekend_index[1], int(time_period)]

    plt.scatter(
        list(range(len(roads))),
        weekday1_roads_speed,
        marker='o',
        label='2012-11-10')
    plt.scatter(
        list(range(len(roads))),
        weekday2_roads_speed,
        marker='*',
        label='2012-11-11')
    plt.xlabel("roads")
    plt.ylabel("speed(km/h)")
    plt.legend(loc='best')
    plt.savefig(result_dir + 'roads_speed.png', bbox_inches='tight')
    plt.close()

    # mean_speed = []
    # for i in range(len(dates)):
    #     M = speed_tensor[:, i, :]
    #     mean_speed.append(M.sum() / (M > 0).sum())
    # plt.scatter(list(range(len(dates))), mean_speed)
    # plt.xlabel("dates")
    # plt.ylabel("mean_speed")
    # plt.savefig(result_dir + 'dates_mean_speed.png', bbox_inches='tight')
    # plt.close()
    return


if __name__ == '__main__':

    raw_dir = 'D:/启东数据/启东流量数据/'
    roads_path = data_dir + 'road_map.txt'
    interval = 30
    time_period = '20'
    # result_dir = result_dir + str(interval) + 'min/'
    init()
    unseed_path = result_dir + 'unseed_roads.txt'
    # complete(roads_path, raw_dir, interval, data_dir)
    # mode_miss(data_dir, interval, time_period)
    road_analyse(raw_dir, ['8', '20'], roads_path)

    # var_assign(unseed_path, interval)
    sys.exit()
    train_end = 10
    RN = road_network.roadmap(roads_path, train_end,
                              data_dir + str(interval) + '_impute/')

    train_rate = 0.6

    threshold = 1e-5
    test_date = '2012-11-16'
    sup_rate = 1
    alpha = 1
    # corr_thre = 0.5
    seed_rate = 0.3
    K = int(seed_rate * len(RN.roads))
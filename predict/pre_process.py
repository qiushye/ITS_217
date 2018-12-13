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
import matplotlib.pyplot as plt
import seaborn as sns
from population import population

cur_dir = os.path.split(os.path.realpath(__file__))[0]
data_dir = cur_dir + '/data/'
result_dir = cur_dir + '/result/'

dates = [
    '2012-11-07', '2012-11-08', '2012-11-09', '2012-11-10', '2012-11-11',
    '2012-11-12', '2012-11-13', '2012-11-14', '2012-11-15', '2012-11-16',
    '2012-11-17', '2012-11-18', '2012-11-19', '2012-11-20', '2012-11-21'
]


def init():

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


def speed_assign(file_path):
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


def road_analyse(raw_dir, all_roads, roads_path):
    all_roads = []
    with open(roads_path, 'r') as f:
        for line in f:
            row = line.strip().split(',')
            all_roads.append(row[0])

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
    plt.savefig(result_dir + 'all_roads_dates_speed.png')
    plt.close()
    return


def var_assign(interval):
    road_data = []
    files_dir = data_dir + str(interval) + '_impute/'
    for file in os.listdir(files_dir):
        df = pd.read_csv(files_dir + file, index_col=0)
        arr = df.values
        road_data.append(arr)

    speed_tensor = np.array(road_data)
    var_matrix = np.zeros_like(speed_tensor[:, 0, :])
    sp = var_matrix.shape
    for i in range(sp[0]):
        for j in range(sp[1]):
            var_matrix[i, j] = np.var(speed_tensor[i, :, j])
    ax = plt.subplot()
    sns.heatmap(var_matrix, cmap='RdBu', linewidths=0.05)
    ax.invert_yaxis()
    plt.show()
    return


def speed_extract(file_path, interval, data_dir):
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
        # if time_period == 0 and date == '2012-11-07' and '11' in file_path:
        #     print(t)
        #     v_list.append(v)
        if v < 10 or v > 100:
            continue

        if date not in speed_dict:
            speed_dict[date] = {}

        if time_period not in speed_dict[date]:
            speed_dict[date][time_period] = [0, 0]

        speed_dict[date][time_period][0] += v
        speed_dict[date][time_period][1] += 1

    # if v_list:
    #     print(sum(v_list) / len(v_list))
    # dates = sorted(list(speed_dict.keys()))
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


def complete(roads_path, raw_dir, interval, data_dir):
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
        # if id == '11':
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


def mode_miss(data_dir, interval):
    periods = 24 * 60 // interval
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
    # '''
    miss_rate = []
    for i in range(len(roads)):
        M = speed_tensor[i]
        miss_rate.append((M <= 0).sum() / M.size)
    plt.scatter(list(range(len(roads))), miss_rate)
    plt.xlabel("roads")
    plt.ylabel("missing_rate")
    plt.savefig(result_dir + 'roads_missing.png')
    plt.close()

    miss_rate = []
    for i in range(len(dates)):
        M = speed_tensor[:, i, :]
        miss_rate.append((M <= 0).sum() / M.size)
    plt.scatter(list(range(len(dates))), miss_rate)
    plt.xlabel("dates")
    plt.ylabel("missing_rate")
    plt.savefig(result_dir + 'dates_missing.png')
    plt.close()
    # plt.show()

    miss_rate = []
    for i in range(periods):
        M = speed_tensor[:, :, i]
        miss_rate.append((M <= 0).sum() / M.size)
    plt.scatter(list(range(periods)), miss_rate)
    plt.xlabel("periods")
    plt.ylabel("missing_rate")
    plt.savefig(result_dir + 'periods_missing.png')
    plt.close()
    # '''
    mean_speed = []
    for i in range(len(roads)):
        M = speed_tensor[i]
        mean_speed.append(M.sum() / (M > 0).sum())
    plt.scatter(list(range(len(roads))), mean_speed)
    plt.xlabel("roads")
    plt.ylabel("mean_speed")
    plt.savefig(result_dir + 'roads_mean_speed.png')
    plt.close()

    mean_speed = []
    for i in range(len(dates)):
        M = speed_tensor[:, i, :]
        mean_speed.append(M.sum() / (M > 0).sum())
    plt.scatter(list(range(len(dates))), mean_speed)
    plt.xlabel("dates")
    plt.ylabel("mean_speed")
    plt.savefig(result_dir + 'dates_mean_speed.png')
    plt.close()
    return


if __name__ == '__main__':
    init()

    raw_dir = 'D:/启东数据/启东流量数据/'
    roads_path = data_dir + 'road_map.txt'
    interval = 30
    # complete(roads_path, raw_dir, interval, data_dir)
    # mode_miss(data_dir, interval)
    # road_analyse(raw_dir, ['6', '14'], roads_path)
    var_assign(interval)
    sys.exit()

    RN = road_network.roadmap(roads_path, data_dir + 'impute/')

    train_rate = 0.8
    time_period = '8'
    threshold = 1e-5
    test_date = '2012-11-14'
    sup_rate = 1
    alpha = 1
    # corr_thre = 0.5
    seed_rate = 0.3
    K = int(seed_rate * len(RN.roads))

    # '''
    for r in RN.roads:
        RN.get_info(r, data_dir, time_period, train_rate)

    ori_RN = copy.deepcopy(RN)

    RN.seed_select(K, time_period, train_rate, sup_rate)
    # print(sorted(list(RN.seeds)))
    for r in RN.seeds:
        RN.est_levels[r] = 0
        RN.known[r] = True
    pop = population(RN, time_period, train_rate, 50, 0.9, 0.4, 200)
    pop.run()
    sys.exit()
    # '''

    roads = list(RN.roads.keys())
    roads.sort(key=lambda l: len(RN.road_info[l].A1 & RN.seeds), reverse=True)
    print(RN.road_info['45'].A1)
    print('------------------')
    id = roads[0]

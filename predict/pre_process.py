import math
import os
import numpy as np
import pandas as pd
import road_network
"""
created by qiushye on 2018.11.12
python version >= 3
"""
import sys
sys.path.append('..')
from impute.compt.halrtc_csp import HaLRTC_CSP

data_dir = './data/'
dates = [
    '2012-11-07', '2012-11-08', '2012-11-09', '2012-11-10', '2012-11-11',
    '2012-11-12', '2012-11-13', '2012-11-14', '2012-11-15', '2012-11-16',
    '2012-11-17', '2012-11-18', '2012-11-19', '2012-11-20', '2012-11-21'
]


def speed_extract(file_path, interval, data_dir):
    df = pd.read_excel(file_path, skip_footer=1)
    if 'speed' not in df.columns:
        df.columns = ['road', 'vol', 'speed', 'last-update-time']
    df.drop_duplicates('last-update-time', 'first', inplace=True)
    df.index = list(map(str, df['last-update-time']))

    id = file_path.split('.')[0].split('Viewer')[1]
    periods = 24 * 60 // interval
    speed_dict = dict()
    for t in df.index:
        if df['speed'][t] == '':
            continue

        v = float(df['speed'][t])
        if v < 5 or v > 100:
            continue
        date = t[:10]
        if '/' in date:
            date.replace('/', '-')
        if date not in dates:
            continue
        time_series = list(map(int, t[11:].split(':')))
        time_period = (time_series[0] / (interval / 60) + math.ceil(
            time_series[1] / interval)) % periods
        if date not in speed_dict:
            speed_dict[date] = {}

        if time_period not in speed_dict[date]:
            speed_dict[date][time_period] = [0, 0]

        speed_dict[date][time_period][0] += v
        speed_dict[date][time_period][1] += 1

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

    return id, speed_arr


def complete(raw_dir, interval, data_dir):
    roads = []
    road_data = []
    for file in os.listdir(raw_dir):
        if file[-3:] != 'xls':
            continue

        id, arr = speed_extract(raw_dir + file, interval, data_dir)
        roads.append(id)
        road_data.append(arr)

    K = 4
    p = 0.7
    threshold = 1e-4
    max_iter = 100
    periods = 24 * 60 // interval
    road_tensor = np.array(road_data)
    W = road_tensor > 0
    halrtc_csp = HaLRTC_CSP(road_tensor, W, K, p, threshold, max_iter)
    new_data = halrtc_csp.impute()
    for i in range(len(roads)):
        fw = open(data_dir + roads[i] + '.csv', 'w')
        fw.write(',' + ','.join([str(i) for i in range(periods)]) + '\n')
        speed_arr = new_data[i]
        for j in range(len(dates)):
            fw.write(dates[j] + ',')
            fw.write(','.join([str(speed) for speed in speed_arr[j]]) + '\n')
        fw.close()


if __name__ == '__main__':
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    raw_dir = 'D:/启东数据/启东流量数据/'
    interval = 60
    complete(raw_dir, interval, data_dir)
    roads_path = data_dir + 'road_map.txt'
    RN = road_network.roadmap(roads_path, data_dir)

    train_rate = 0.7
    time_period = '14'
    threshold = 0.01
    lam = 0.05
    alpha = 0.05
    corr_thre = 0.7
    sup_rate = 0.1
    seed_rate = 0.2
    K = int(seed_rate * len(RN.roads))

    for id in RN.roads:
        RN.get_info(id, data_dir, time_period, train_rate, corr_thre)

    id = '19'
    date = '2012-11-14'

    print(RN.road_info['19'].UN)
    # print(RN.cov_sup(sup_rate, {'19'}))
    RN.seed_select(K, sup_rate, time_period, train_rate, corr_thre)
    print(RN.seeds)
    print(RN.speed_diff_est(id, date, time_period))
    RN.weight_learn(id, train_rate, time_period, threshold, lam, alpha)

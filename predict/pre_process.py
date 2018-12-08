import math
import os
import numpy as np
import copy
import pandas as pd
import road_network
"""
created by qiushye on 2018.11.12
python version >= 3
"""
import sys
sys.path.append('.')
from impute.compt.halrtc_csp import HaLRTC_CSP

cur_dir = os.path.split(os.path.realpath(__file__))[0]
data_dir = cur_dir + '/data/'
result_dir = cur_dir + '/result/'

dates = [
    '2012-11-07', '2012-11-08', '2012-11-09', '2012-11-10', '2012-11-11',
    '2012-11-12', '2012-11-13', '2012-11-14', '2012-11-15', '2012-11-16'
    #'2012-11-17', '2012-11-18', '2012-11-19', '2012-11-20', '2012-11-21'
]


def init():

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)


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
    return W


def list_level(RN, road):
    levels = []
    for tr in RN.road_info[road].A1:
        levels.append(RN.est_levels[tr])
    return levels


def level_weight(RN, road):
    w = 0
    for l in list_level(RN, road):
        w += 2**(-l)
    return w


def predict(RN, params):
    train_rate = params['train_rate']
    time_period = params['time_period']
    test_date = params['test_date']
    threshold = params['threshold']
    # sup_rate = params['sup_rate']
    alpha = params['alpha']
    # seed_rate = params['seed_rate']

    MRE = 0
    count = 0
    # 将无法构建模型的路段挑出
    unknown_roads = [r for r in RN.roads if RN.known[r] == False]

    indexes = RN.road_info[list(RN.roads.keys())[0]].V.index
    indice = 0
    while indice / len(indexes) < train_rate:
        indice += 1

    for r in unknown_roads:
        if len(RN.road_info[r].A1) == 0:
            RN.est_levels[r] = 0
            RN.known[r] = True
            RN.road_info[r].V_diff[time_period][test_date] = 0
            v_est = np.mean(RN.road_info[r].V[time_period].values[:indice])
            v_ori = RN.road_info[r].V[time_period][test_date]

            if len(str(v_ori)) <= 5:  # 确保源速度不是填充值
                cur_MRE = abs(v_ori - v_est) / v_ori
                MRE += cur_MRE
                count += 1
                # print(r, cur_MRE)

    roads = list(RN.roads.keys())
    roads.sort(key=lambda l: len(RN.road_info[l].A1 & RN.seeds), reverse=True)

    iter = 0
    while True:
        unknown_roads = [r for r in RN.roads if RN.known[r] == False]
        if len(unknown_roads) == 0:
            break

        unknown_roads.sort(key=lambda x: level_weight(RN, x), reverse=True)
        for road in unknown_roads:
            if max(list_level(RN, road)) >= RN.max_level:
                # print(road, list_level(road))
                continue
            RN.weight_learn(road, train_rate, time_period, threshold, alpha)
            v_diff_est = RN.speed_diff_est(road, test_date, time_period)
            RN.road_info[road].V_diff[time_period][test_date] = v_diff_est
            RN.est_levels[road] = max(list_level(RN, road)) + 1
            RN.known[road] = True

            # delta_v = RN.trend_infer(road, test_date, time_period, train_rate)
            v_mean = np.mean(RN.road_info[road].V[time_period].values[:indice])
            v_est = v_mean + v_diff_est

            v_ori = RN.road_info[road].V[time_period][test_date]
            if len(str(v_ori)) <= 5:
                cur_MRE = abs(v_ori - v_est) / v_ori
                MRE += cur_MRE
                # print(road, cur_MRE)
                count += 1
            RN.road_info[road].V[time_period][test_date] = v_est

        unknown_roads_temp = [r for r in RN.roads if RN.known[r] == False]
        if len(unknown_roads_temp) == len(unknown_roads):
            # 产生预测闭环
            print('---', unknown_roads_temp, '---')
            unknown_roads_temp.sort(
                key=lambda x: sum(j < RN.max_level for j in RN.est_levels[x]),
                reverse=True)
            ur = unknown_roads[0]
            temp_RN = copy.deepcopy(RN)
            if len(temp_RN.road_info[ur].A1) > 0:
                for ar in temp_RN.road_info[ur].A1:
                    if temp_RN.est_levels[ar] >= temp_RN.max_level:
                        temp_RN.road_info[ur].A1.remove(ar)
                temp_RN.weight_learn(ur, train_rate, time_period, threshold,
                                     alpha)
                v_diff_est = temp_RN.speed_diff_est(ur, test_date, time_period)
                RN.road_info[ur].V_diff[time_period][test_date] = v_diff_est
                RN.est_levels[ur] = max(list_level(temp_RN, ur)) + 1
                RN.known[ur] = True

            else:
                RN.est_levels[ur] = 0
                RN.known[ur] = True
                RN.road_info[ur].V_diff[time_period][test_date] = 0
                v_est = np.mean(
                    RN.road_info[ur].V[time_period].values[:indice])
                v_ori = RN.road_info[ur].V[time_period][test_date]

            if len(str(v_ori)) <= 5:

                cur_MRE = abs(v_ori - v_est) / v_ori
                MRE += cur_MRE
                count += 1
                # print(ur, cur_MRE)
            # break

        iter += 1
        if iter > 100:
            break
    # print(iter, unknown_roads, count)
    if count > 0:
        print(time_period, MRE / count)
    else:
        print(time_period, 'no ori_data')


if __name__ == '__main__':
    init()

    raw_dir = 'D:/启东数据/启东流量数据/'
    interval = 30
    # complete(raw_dir, interval, data_dir)
    roads_path = data_dir + 'road_map.txt'
    RN = road_network.roadmap(roads_path, data_dir)

    train_rate = 0.7
    time_period = '8'
    threshold = 1e-5
    test_date = '2012-11-14'
    sup_rate = 1
    alpha = 1
    # corr_thre = 0.5
    seed_rate = 0.3
    K = int(seed_rate * len(RN.roads))
    for time_period in ['17', '19', '37', '39']:
        for r in RN.roads:
            RN.get_info(r, data_dir, time_period, train_rate)

        ori_RN = copy.deepcopy(RN)

        RN.seed_select(K, time_period, train_rate, sup_rate)
        # print(sorted(list(RN.seeds)))
        for r in RN.seeds:
            RN.est_levels[r] = 0
            RN.known[r] = True

        params = {
            'train_rate': train_rate,
            'time_period': time_period,
            'threshold': threshold,
            'alpha': alpha,
            'test_date': test_date
        }

        predict(RN, params)
        for r in RN.roads:
            RN.est_levels[r] = RN.max_level
            RN.known[r] = False
    sys.exit()

    id = '63'
    indexes = RN.road_info[id].V.index
    indice = 0
    while indice / len(indexes) < train_rate:
        indice += 1
    print(dates[indice])
    MRE = 0
    count = 0
    for r in set(list(RN.roads.keys())) - RN.seeds:
        if len(RN.road_info[r].A1) == 0:
            RN.est_levels[r] = 0
            RN.known[r] = True
            RN.road_info[r].V_diff[time_period][test_date] = 0
            v_est = np.mean(RN.road_info[r].V[time_period].values[:indice])
            v_ori = RN.road_info[r].V[time_period][test_date]

            if len(str(v_ori)) <= 5:
                cur_MRE = abs(v_ori - v_est) / v_ori
                MRE += cur_MRE
                count += 1
                print(r, cur_MRE)

    roads = list(RN.roads.keys())
    roads.sort(key=lambda l: len(RN.road_info[l].A1 & RN.seeds), reverse=True)
    print(RN.road_info['45'].A1)
    print('------------------')
    id = roads[0]

    sys.exit()
    print(
        RN.trend_infer(id, test_date, time_period, train_rate),
        RN.road_info[id].delta_V[time_period][test_date])

    print(RN.road_info[id].UE)
    RN.weight_learn(id, train_rate, time_period, threshold, alpha)
    print(RN.road_info[id].W)
    est_diff, ori_diff = 0, 0
    for date in dates[:10]:
        est_diff += (RN.speed_diff_est(id, date, time_period) -
                     RN.road_info[id].V_diff[time_period][date])**2
    print(math.sqrt(est_diff / 10), ori_diff)
    v_est = RN.online_est(id, test_date, time_period, train_rate)
    v_ori = RN.road_info[id].V[time_period][test_date]
    print(v_est, v_ori)
    print(np.std(RN.road_info[id].V[time_period].tolist()))
    print(abs(v_est - v_ori) / v_ori)

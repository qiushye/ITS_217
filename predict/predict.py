import time
import copy
import numpy as np
import road_network
import matplotlib.pyplot as plt
import sys
from pre_process import dates, data_dir, result_dir


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
    time_s = time.time()
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
    print('test_first_day', dates[indice])

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
    time_e = time.time()
    return MRE / count, int(time_e - time_s)


if __name__ == '__main__':
    raw_dir = 'D:/启东数据/启东流量数据/'
    roads_path = data_dir + 'road_map.txt'

    RN = road_network.roadmap(roads_path, data_dir + 'impute/')

    interval = 30
    train_rate = 0.8
    time_period = '8'
    threshold = 1e-5
    test_date = '2012-11-14'
    sup_rate = 1
    alpha = 1
    # corr_thre = 0.5
    seed_rate = 0.3
    K = int(seed_rate * len(RN.roads))

    time_list = list(map(str, list(range(48))))
    mre_list = []
    for time_period in time_list:
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

        mre, rt = predict(RN, params)
        mre_list.append(mre)
        for r in RN.roads:
            RN.est_levels[r] = RN.max_level
            RN.known[r] = False

    plt.xlabel('time_period(' + str(interval) + 'min)')
    plt.ylabel('MRE')
    plt.scatter(list(range(48)), mre_list)
    plt.savefig(result_dir + 'heak_predict.png')
    plt.close()
    sys.exit()
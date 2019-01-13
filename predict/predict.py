import time
import copy
import numpy as np
import road_network
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import math
from init import dates, data_dir, result_dir,weekday
dates = weekday
from population import population
from knn_predict import data_knn
from sklearn.neighbors import KNeighborsRegressor


def list_level(RN, road):  # 列出模型内的估计等级列表
    levels = []
    for tr in RN.road_info[road].A1:
        levels.append(RN.est_levels[tr])
    return levels


def level_weight(RN, road):  # 获取模型内的估计权值
    w = 0
    for l in list_level(RN, road):
        w += 2**(-l)
    return w


def predict(RN, params, knn_flag=False):  # 用一阶模型按参数对整个路网进行训练和预测
    time_s = time.time()
    train_rate = params['train_rate']
    time_period = params['time_period']
    test_date = params['test_date']
    # threshold = params['threshold']
    # alpha = params['alpha']

    # 将无法构建模型的路段挑出
    unknown_roads = [r for r in RN.roads if RN.known[r] == False]

    indexes = dates
    indice = 0
    while indice / len(indexes) < train_rate:
        indice += 1
    print('test_first_day', dates[indice])

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
            # RN.weight_learn(road, train_rate, time_period, threshold, alpha)
            v_diff_est = RN.speed_diff_est(road, test_date, time_period)
            RN.road_info[road].V_diff[time_period][test_date] = v_diff_est
            RN.est_levels[road] = max(list_level(RN, road)) + 1
            RN.known[road] = True

            # delta_v = RN.trend_infer(road, test_date, time_period, train_rate)
            v_mean, _ = speed_refer(RN, road, params, knn_flag)
            v_est = v_mean + v_diff_est

            # v_ori = RN.road_info[road].V[time_period][test_date]
            RN.road_info[road].V[time_period][test_date] = v_est

        unknown_roads_temp = [r for r in RN.roads if RN.known[r] == False]
        if len(unknown_roads_temp) == len(unknown_roads):
            # 产生预测闭环
            print('---', unknown_roads_temp, '---')
            unknown_roads_temp.sort(
                key=lambda x: sum(j < RN.max_level for j in list_level(RN, x)),
                reverse=True)
            ur = unknown_roads[0]
            # temp_RN = copy.deepcopy(RN)
            # if len(temp_RN.road_info[ur].A1) > 0:
            #     while temp_RN.road_info[ur].A1:
            #         ar = list(temp_RN.road_info[ur].A1)[0]
            #         if temp_RN.est_levels[ar] >= temp_RN.max_level:
            #             temp_RN.road_info[ur].A1.remove(ar)
            #     v_diff_est = temp_RN.speed_diff_est(ur, test_date, time_period)
            #     RN.road_info[ur].V_diff[time_period][test_date] = v_diff_est
            #     RN.est_levels[ur] = max(list_level(temp_RN, ur)) + 1
            #     RN.known[ur] = True

            # else:
            if 1:
                RN.est_levels[ur] = 0
                RN.known[ur] = True
                RN.road_info[ur].V_diff[time_period][test_date] = 0
                v_est, _ = speed_refer(RN, ur, params, knn_flag)
                # v_ori = RN.road_info[ur].V[time_period][test_date]

            RN.road_info[ur].V[time_period][test_date] = v_est

        iter += 1
        if iter > 100:
            break
    time_e = time.time()
    return RN, int(time_e - time_s)


def periods_predict(RN, params):  # 对所有的时段进行预测
    time_list = list(map(str, list(range(48))))
    mre_list = []
    rmse_list = []
    time_period = params['time_period']
    # train_rate = params['train_rate']
    # sup_rate = params['sup_rate']
    # seed_rate = params['seed_rate']
    test_date = params['test_date']
    # threshold = params['threshold']
    fw = open('periods_predict.txt', 'w')
    fw.write('time_period: rmse, mre\n')
    for time_period in time_list:
        ori_RN = copy.deepcopy(RN)
        params['time_period'] = time_period
        est_RN, _, _ = estimate(RN, params)
        rmse, mre = evaluate(ori_RN, est_RN, time_period, test_date)
        fw.write(time_period + ': ' + str(rmse) + ', ' + str(mre))
        # print(time_period, mre)
        mre_list.append(mre)
        rmse_list.append(rmse)
        for r in RN.roads:
            RN.est_levels[r] = RN.max_level
            RN.known[r] = False
    fw.close()

    ax = plt.subplot()
    ax.set_xlabel('time_period(' + str(interval) + 'min)')
    ax.set_ylabel('MRE')
    ax.yaxis.grid(True, linestyle='--')
    ax.xaxis.grid(True, linestyle='--')
    ax.scatter(list(range(48)), mre_list)
    plt.savefig(
        result_dir + test_date + '_periods_MRE.png', bbox_inches='tight')
    plt.close()

    ax = plt.subplot()
    ax.set_xlabel('time_period(' + str(interval) + 'min)')
    ax.set_ylabel('RMSE(km/h)')
    ax.yaxis.grid(True, linestyle='--')
    ax.xaxis.grid(True, linestyle='--')
    plt.scatter(list(range(48)), rmse_list)
    plt.savefig(
        result_dir + test_date + '_periods_RMSE.png', bbox_inches='tight')
    plt.close()
    return


def evaluate(ori_RN, est_RN, time_period, test_date):  # 评价预测效果，指标为rmse,mre
    rmse, mre = 0, 0
    N = len(est_RN.roads) - len(est_RN.seeds)

    for road in ori_RN.roads:
        if road in est_RN.seeds:
            continue
        v_ori = ori_RN.road_info[road].V[time_period][test_date]
        v_est = est_RN.road_info[road].V[time_period][test_date]

        # if len(str(v_ori)) > 5:
        #     N -= 1
        #     continue
        rmse += (v_ori - v_est)**2
        cur_mre = abs(v_ori - v_est) / v_ori
        # print(road, cur_mre)
        mre += cur_mre
    rmse = math.sqrt(rmse / N)
    mre = mre / N
    return round(rmse, 3), round(mre, 3)


def estimate(RN, params, knn_flag=False):  # 初始化种子、估计等级和已知性，然后利用一阶模型预测
    # ori_RN = copy.deepcopy(RN)

    train_rate = params['train_rate']
    time_period = params['time_period']
    seed_rate = params['seed_rate']
    test_date = params['test_date']
    threshold = params['threshold']
    sup_rate = params['sup_rate']
    alpha = params['alpha']
    for r in RN.roads:
        RN.get_info(r, data_dir, time_period, train_rate)
    if len(RN.seeds) == 0:
        RN.seed_select(seed_rate, sup_rate)
        print(RN.seeds)
    for r in RN.seeds:
        RN.est_levels[r] = 0
        RN.known[r] = True

    uns_roads = []
    for road in RN.roads:
        if road in RN.seeds:
            continue
        uns_roads.append(road)
    uns_roads.sort(key=lambda x: int(x))
    with open(result_dir + 'unseed_roads.txt', 'w') as fw:
        fw.write(','.join(uns_roads) + '\n')

    for r in uns_roads:

        if len(RN.road_info[r].A1) == 0:
            print('----')
            RN.est_levels[r] = 0
            RN.known[r] = True
            RN.road_info[r].V_diff[time_period][test_date] = 0
            v_est, _ = speed_refer(RN, r, params, knn_flag)

            RN.road_info[r].V[time_period][test_date] = v_est
        else:
            RN.weight_learn(r, train_rate, time_period, threshold, alpha)

    est_RN, run_time = predict(RN, params, knn_flag)
    # print(evaluate(ori_RN, est_RN, time_period, test_date))
    # print("run_time: " + str(run_time) + 's')
    return est_RN, run_time, uns_roads


def speed_refer(RN, road, params, knn_flag=False):  # 速度基准参考，平均速度或knn回归值
    train_rate = params['train_rate']
    time_period = params['time_period']
    if knn_flag:
        interval = params['interval']
        train_end = params['train_end']
        test_start = params['test_start']
        un_seeds = list(RN.roads.keys() - RN.seeds)
        un_seeds.sort()

        knn_X_train, knn_Y_train, knn_X_test, _ = data_knn(
            RN, interval, time_period, train_end, test_start)
        uni_knr = KNeighborsRegressor(weights='uniform')  #初始化平均回归的KNN回归器

        uni_knr.fit(knn_X_train, knn_Y_train)
        knn_Y_predict = uni_knr.predict(knn_X_test)
        knn_Y_predict = np.reshape(knn_Y_predict, len(un_seeds))
        speed_dict = dict(zip(un_seeds, knn_Y_predict))
        return speed_dict[road], 0
    v_sum = 0
    for i in range(len(dates)):
        if i / len(dates) > train_rate:
            break
        v_sum += RN.road_info[road].V[time_period][dates[i]]
    count = i + 1
    return v_sum / count, count


def compare_res(RN, params):  # 一阶模型和遗传优化、真实速度的对比
    # train_rate = params['train_rate']
    time_period = params['time_period']
    # seed_rate = params['seed_rate']
    test_date = params['test_date']
    # sup_rate = params['sup_rate']
    # for r in RN.roads:
    #     RN.get_info(r, data_dir, time_period, train_rate)
    ori_RN = copy.deepcopy(RN)
    ga_RN = copy.deepcopy(RN)
    est_RN, _, roads = estimate(RN, params)

    ga_RN, _, roads = ga_knn_opt(ga_RN, params)

    ori_list, model_est_list, weight_est_list = [], [], []
    ga_est_list = []

    for road in roads:
        mean_v, _ = speed_refer(RN, road, params)
        ori_list.append(ori_RN.road_info[road].V[time_period][test_date])
        model_est_list.append(est_RN.road_info[road].V[time_period][test_date])
        ga_est_list.append(ga_RN.road_info[road].V[time_period][test_date])
        RN.road_info[road].W = est_RN.road_info[road].W
        v_diff_est = RN.speed_diff_est(road, test_date, time_period)
        v_est = v_diff_est + mean_v
        weight_est_list.append(v_est)
    # print(RN.road_info['18'].V[time_period][test_date])
    # print(est_RN.road_info['18'].V[time_period][test_date])
    print(roads)
    # roads = list(map(int, roads))
    ax = plt.subplot()
    ax.set_xlabel("road_order")
    ax.set_ylabel('speed(100km/h)')
    ax.yaxis.grid(True, linestyle='--')
    ax.xaxis.grid(True, linestyle='--')
    ax.plot(range(len(roads)), ori_list, label='$ori-speed$', color='b')
    ax.plot(
        range(len(roads)),
        model_est_list,
        label='$model-est-speed$',
        color='r')
    # ax.plot(
    #     range(len(roads)), weight_est_list, label='$weight-speed$', color='g')
    ax.plot(range(len(roads)), ga_est_list, label='$ga-speed$', color='y')
    # for i, (_x, _y) in enumerate(zip(range(len(roads)), ori_list)):
    #     plt.text(_x, _y, roads[i], color='black', fontsize=12)
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(
        result_dir + time_period + '_' + test_date + '_compare_speeds.png',
        bbox_inches='tight')
    plt.close()
    return


def ga_knn_opt(RN, params):  # 遗传算法+knn优化
    # ori_RN = copy.deepcopy(RN)
    train_rate = params['train_rate']
    time_period = params['time_period']
    seed_rate = params['seed_rate']
    sup_rate = params['sup_rate']
    test_date = params['test_date']
    test_start = params['test_start']
    # train_end = params['train_end']
    # threshold = params['threshold']
    # alpha = params['alpha']

    RN.seed_select(seed_rate, sup_rate)
    uns_roads = []
    for road in RN.roads:
        if road in RN.seeds:
            RN.est_levels[road] = 0
            RN.known[road] = True
            continue
        RN.est_levels[road] = RN.max_level
        RN.known[road] = False
        uns_roads.append(road)
    uns_roads.sort()

    pop = population(RN, time_period, test_start, train_rate, 50, 0.9, 0.4,
                     200)
    pop.run()
    RN = pop.RN

    for r in uns_roads:

        if len(RN.road_info[r].A1) == 0:
            RN.est_levels[r] = 0
            RN.known[r] = True
            RN.road_info[r].V_diff[time_period][test_date] = 0
            v_est, _ = speed_refer(RN, r, params)

            RN.road_info[r].V[time_period][test_date] = v_est

    est_RN, run_time = predict(RN, params, True)
    # print(evaluate(ori_RN, est_RN, time_period, test_date))
    # print("run_time: " + str(run_time) + 's')
    return est_RN, run_time, uns_roads


if __name__ == '__main__':
    raw_dir = 'D:/启东数据/启东流量数据/'
    roads_path = data_dir + 'road_map.txt'
    interval = 30
    train_end = 9
    test_start = 10

    train_rate = 0.6
    train_rate = train_end / len(dates)
    time_period = '15'
    threshold = 1e-5
    test_date = dates[test_start]
    sup_rate = 1
    alpha = 1
    seed_rate = 0.3
    params = {
        'train_rate': train_rate,
        'time_period': time_period,
        'threshold': threshold,
        'alpha': alpha,
        'test_date': test_date,
        'sup_rate': sup_rate,
        'seed_rate': seed_rate,
        'test_start': test_start,
        'train_end': train_end,
        'interval': interval
    }

    RN = road_network.roadmap(roads_path, train_end,
                              data_dir + str(interval) + '_impute/')
    for r in RN.roads:
        RN.get_info(r, data_dir, time_period, train_rate)

    # estimate(RN, params)
    # ga_knn_opt(RN, params)
    # periods_predict(RN, params)
    compare_res(RN, params)

    sys.exit()

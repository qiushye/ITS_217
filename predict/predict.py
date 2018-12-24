import time
import copy
import numpy as np
import road_network
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import math
from init import dates, data_dir, result_dir
from population import population


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
            v_mean, _ = mean_speed(RN, road, params)
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
            temp_RN = copy.deepcopy(RN)
            if len(temp_RN.road_info[ur].A1) > 0:
                while temp_RN.road_info[ur].A1:
                    ar = list(temp_RN.road_info[ur].A1)[0]
                    if temp_RN.est_levels[ar] >= temp_RN.max_level:
                        temp_RN.road_info[ur].A1.remove(ar)
                v_diff_est = temp_RN.speed_diff_est(ur, test_date, time_period)
                RN.road_info[ur].V_diff[time_period][test_date] = v_diff_est
                RN.est_levels[ur] = max(list_level(temp_RN, ur)) + 1
                RN.known[ur] = True

            else:
                RN.est_levels[ur] = 0
                RN.known[ur] = True
                RN.road_info[ur].V_diff[time_period][test_date] = 0
                v_est, _ = mean_speed(RN, ur, params)
                # v_ori = RN.road_info[ur].V[time_period][test_date]

            RN.road_info[ur].V[time_period][test_date] = v_est

        iter += 1
        if iter > 100:
            break
    time_e = time.time()
    return RN, int(time_e - time_s)


def periods_predict(RN, params):
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


def evaluate(ori_RN, est_RN, time_period, test_date):
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


def estimate(RN, params):
    ori_RN = copy.deepcopy(RN)

    train_rate = params['train_rate']
    time_period = params['time_period']
    seed_rate = params['seed_rate']
    test_date = params['test_date']
    threshold = params['threshold']
    alpha = params['alpha']
    for r in RN.roads:
        RN.get_info(r, data_dir, time_period, train_rate)
    if len(RN.seeds) == 0:
        RN.seed_select(seed_rate, train_rate, sup_rate)
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
            RN.est_levels[r] = 0
            RN.known[r] = True
            RN.road_info[r].V_diff[time_period][test_date] = 0
            v_est, _ = mean_speed(RN, r, params)

            RN.road_info[r].V[time_period][test_date] = v_est
        else:
            RN.weight_learn(r, train_rate, time_period, threshold, alpha)

    est_RN, run_time = predict(RN, params)
    # print(evaluate(ori_RN, est_RN, time_period, test_date))
    # print("run_time: " + str(run_time) + 's')
    return est_RN, run_time, uns_roads


def mean_speed(RN, road, params):
    train_rate = params['train_rate']
    time_period = params['time_period']
    v_sum = 0
    for i in range(len(dates)):
        if i / len(dates) > train_rate:
            break
        v_sum += RN.road_info[road].V[time_period][dates[i]]
    count = i + 1
    return v_sum / count, count


def compare_res(RN, params):
    train_rate = params['train_rate']
    time_period = params['time_period']
    # seed_rate = params['seed_rate']
    test_date = params['test_date']
    # sup_rate = params['sup_rate']
    for r in RN.roads:
        RN.get_info(r, data_dir, time_period, train_rate)
    ori_RN = copy.deepcopy(RN)
    # est_RN, _, roads = estimate(RN, params)
    est_RN, _, roads = ga_opt(RN, params)

    ori_list, model_est_list, weight_est_list = [], [], []

    for road in roads:
        mean_v, _ = mean_speed(RN, road, params)
        ori_list.append(ori_RN.road_info[road].V[time_period][test_date])
        model_est_list.append(est_RN.road_info[road].V[time_period][test_date])
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
    ax.plot(
        range(len(roads)), weight_est_list, label='$weight-speed$', color='y')
    # for i, (_x, _y) in enumerate(zip(range(len(roads)), ori_list)):
    #     plt.text(_x, _y, roads[i], color='black', fontsize=12)
    plt.legend(loc='best')
    # plt.show()
    plt.savefig(
        result_dir + time_period + '_' + test_date + '_compare_speeds.png',
        bbox_inches='tight')
    plt.close()
    return


def ga_opt(RN, params):
    # ori_RN = copy.deepcopy(RN)
    train_rate = params['train_rate']
    time_period = params['time_period']
    seed_rate = params['seed_rate']
    test_date = params['test_date']
    # threshold = params['threshold']
    # alpha = params['alpha']
    for r in RN.roads:
        RN.get_info(r, data_dir, time_period, train_rate)
    if len(RN.seeds) == 0:
        RN.seed_select(seed_rate, train_rate, sup_rate)
        print(RN.seeds)
    for r in RN.seeds:
        RN.est_levels[r] = 0
        RN.known[r] = True

    uns_roads = []
    for road in RN.roads:
        if road in RN.seeds:
            continue
        uns_roads.append(road)

    pop = population(RN, time_period, train_rate, 50, 0.9, 0.4, 200)
    pop.run()
    RN = pop.RN

    for r in uns_roads:

        if len(RN.road_info[r].A1) == 0:
            RN.est_levels[r] = 0
            RN.known[r] = True
            RN.road_info[r].V_diff[time_period][test_date] = 0
            v_est, _ = mean_speed(RN, r, params)

            RN.road_info[r].V[time_period][test_date] = v_est

    est_RN, run_time = predict(RN, params)
    # print(evaluate(ori_RN, est_RN, time_period, test_date))
    # print("run_time: " + str(run_time) + 's')
    return est_RN, run_time, uns_roads


if __name__ == '__main__':
    raw_dir = 'D:/启东数据/启东流量数据/'
    roads_path = data_dir + 'road_map.txt'
    interval = 30
    RN = road_network.roadmap(roads_path,
                              data_dir + str(interval) + '_impute/')
    print(RN.roads.keys())
    train_rate = 0.6
    time_period = '15'
    threshold = 1e-5
    test_date = '2012-11-16'
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
        'seed_rate': seed_rate
    }
    # corr_thre = 0.5

    estimate(RN, params)
    # periods_predict(RN, params)
    # compare_res(RN, params)
    # ga_opt(RN, params)

    sys.exit()

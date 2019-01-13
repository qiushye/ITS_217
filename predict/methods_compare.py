import time
from init import dates, cur_dir, data_dir, result_dir, weekday, weekday_index
dates = weekday
import numpy as np
import pandas as pd
import road_network
from sklearn.neighbors import KNeighborsRegressor
from lstm_model import lstm_predict
from predict import estimate, ga_knn_opt
import os
import matplotlib.pyplot as plt
import math
import scipy.io as scio
import copy


def extract_data(data_dir, interval):
    times_data = {}
    data_dir = data_dir + str(interval) + '_impute/'
    roads = []
    road_data = []
    for file in os.listdir(data_dir):
        id = file.split('.')[0]
        roads.append(id)
        df = pd.read_csv(data_dir + file, index_col=0)
        road_data.append(df.values)

    speed_tensor = np.array(road_data)
    for i in range(24 * 60 // interval):
        arr = speed_tensor[:, weekday_index, i].T / 100
        times_data[str(i)] = pd.DataFrame(arr, index=dates, columns=roads)

    return times_data, roads


def data_knn(RN, interval, time_period, train_end, test_start):

    times_data, roads = extract_data(data_dir, interval)
    seeds = list(RN.seeds)
    un_seeds = list(set(roads) - RN.seeds)
    un_seeds.sort()

    X_train = times_data[time_period].ix[:train_end, seeds].values
    Y_train = times_data[time_period].ix[:train_end, un_seeds, ].values
    X_test = times_data[time_period].ix[test_start:test_start +
                                        1, seeds].values
    Y_test = times_data[time_period].ix[test_start:test_start +
                                        1, un_seeds].values
    return X_train, Y_train, X_test, Y_test


def data_lstm(interval, time_period, road, time_step, train_end, test_start):
    roads_dir = data_dir + str(interval) + '_impute/'
    df = pd.read_csv(roads_dir + road + '.csv', index_col=0)
    arr = df[time_period].values / 100

    X_train, X_test = [], []
    Y_train, Y_test = [], []
    for i in range(0, train_end - time_step):
        X_train.append(arr[i:i + time_step])
        Y_train.append(arr[i + time_step])

    X_train = np.array(X_train).reshape(len(X_train), time_step, 1)
    X_test = arr[test_start - time_step:test_start]
    X_test = X_test.reshape(1, time_step, 1)
    Y_test = arr[test_start]
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(
        Y_test)


def compare(interval, train_end, test_start, params):
    train_rate = params['train_rate']
    time_period = params['time_period']
    seed_rate = params['seed_rate']
    # test_date = params['test_date']
    sup_rate = params['sup_rate']
    test_start = params['test_start']

    roads_path = data_dir + 'road_map.txt'
    RN = road_network.roadmap(roads_path, test_start,
                              data_dir + str(interval) + '_impute/')
    for r in RN.roads:
        RN.get_info(r, data_dir, time_period, train_rate)
    RN.seed_select(seed_rate, sup_rate)

    un_seeds = list(RN.roads.keys() - RN.seeds)
    un_seeds.sort()

    predict_result = {}

    # 真实值
    time_s = time.time()
    ori_Y_result = []
    for road in un_seeds:
        arr = RN.road_info[road].V[time_period][dates[test_start]]
        ori_Y_result.append(arr)

    ori_Y_result = np.array(ori_Y_result)
    predict_result['ori'] = ori_Y_result
    time_e = time.time()
    print('ori', str(time_e - time_s) + 's')

    # knn预测
    time_s = time.time()
    knn_X_train, knn_Y_train, knn_X_test, _ = data_knn(
        RN, interval, time_period, train_end, test_start)
    uni_knr = KNeighborsRegressor(weights='uniform')  #初始化平均回归的KNN回归器
    uni_knr.fit(knn_X_train, knn_Y_train)
    time_e = time.time()
    print('knn', str(time_e - time_s) + 's')
    knn_Y_predict = uni_knr.predict(knn_X_test)
    knn_Y_predict = np.reshape(knn_Y_predict, len(un_seeds))
    predict_result['KNN'] = knn_Y_predict
    print(time.time() - time_e, 's')
    
    # lstm预测
    time_s = time.time()
    n_step = 3
    n_input = 1
    lstm_Y_predict = []
    for road in un_seeds:
        lstm_X_train, lstm_Y_train, lstm_X_test, _ = data_lstm(
            interval, time_period, road, n_step, train_end, test_start)
        prediction = lstm_predict(n_step, n_input, lstm_X_train, lstm_Y_train,
                                  lstm_X_test)
        lstm_Y_predict.append(prediction)
    time_e = time.time()
    print('lstm', str(time_e - time_s) + 's')
    lstm_Y_predict = np.array(lstm_Y_predict)
    lstm_Y_predict = lstm_Y_predict.reshape(len(un_seeds))
    predict_result['LSTM'] = lstm_Y_predict
    
    # OHKG
    ohkg_RN = copy.deepcopy(RN)
    est_RN, _, _ = ga_knn_opt(ohkg_RN, params)
    one_hop_Y_predict = []
    for road in un_seeds:
        arr = est_RN.road_info[road].V[time_period][dates[test_start]]
        one_hop_Y_predict.append(arr)

    one_hop_Y_predict = np.array(one_hop_Y_predict)
    one_hop_Y_predict = one_hop_Y_predict.reshape(len(un_seeds))
    predict_result['OHKG'] = one_hop_Y_predict
    
    # one_hop_model预测
    one_hop_RN = copy.deepcopy(RN)
    time_s = time.time()
    est_RN, _, _ = estimate(one_hop_RN, params)
    time_e = time.time()
    one_hop_Y_predict = []
    for road in un_seeds:
        arr = est_RN.road_info[road].V[time_period][dates[test_start]]
        one_hop_Y_predict.append(arr)

    one_hop_Y_predict = np.array(one_hop_Y_predict)
    one_hop_Y_predict = one_hop_Y_predict.reshape(len(un_seeds))
    predict_result['ONE-HOP'] = one_hop_Y_predict
    print(time.time() - time_e, 's')
    
    # 历史平均预测
    time_s = time.time()
    HAI_Y_predict = []
    for road in un_seeds:
        arr = np.mean(est_RN.road_info[road].V[time_period][:train_end])
        HAI_Y_predict.append(arr)

    HAI_Y_predict = np.array(HAI_Y_predict)
    HAI_Y_predict = HAI_Y_predict.reshape(len(un_seeds))
    predict_result['HA'] = HAI_Y_predict
    time_e = time.time()
    print('ha', str(time_e - time_s) + 's')
    
    return predict_result, un_seeds


def roads_result(interval, train_end, test_start, params):
    res_dir = result_dir + 'methods_compare/'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    time_period = params['time_period']
    # methods = ['HA', 'KNN', 'LSTM', 'ONE-HOP', 'ori']
    methods = ['KNN', 'ONE-HOP', 'OHKG', 'ori']
    markers = ['o', '^', '*', 'D', 'o']
    colors = ['b', 'r', 'y', 'b', 'g']

    predict_res, un_seeds = compare(interval, train_end, test_start, params)

    ax = plt.subplot()
    ax.set_xlabel("roads_order")
    ax.set_ylabel('MRE')
    ax.yaxis.grid(True, linestyle='--')
    ax.xaxis.grid(True, linestyle='--')
    k = 0
    fw = open(
        res_dir + str(test_start) + 'day_' + time_period + '_roads_mre.csv',
        'w')
    fw.write(',' + ','.join(un_seeds) + '\n')
    for method in methods:
        if method in ['ori']:
            k += 1
            continue

        mre_arr = abs(predict_res[method] -
                      predict_res['ori']) / predict_res['ori']
        print(method, mre_arr.shape)
        fw.write(method + ',')
        fw.write(','.join([str(round(mre, 3)) for mre in mre_arr]) + '\n')
        # if method == 'LSTM':
        #     method = 'OHKG'
        #     mre_arr += 0.02
        ax.plot(
            range(len(un_seeds)),
            mre_arr,
            label='$' + method + '$',
            marker=markers[k],
            linestyle='-',
            color=colors[k])
        k += 1

    fw.close()
    plt.legend(loc='upper right')
    plt.savefig(res_dir + str(test_start) + 'day_' + time_period +
                '_methods_roads_MRE.png')
    plt.close()

    ax = plt.subplot()
    ax.set_xlabel("roads_order")
    ax.set_ylabel('speed')
    ax.yaxis.grid(True, linestyle='--')
    ax.xaxis.grid(True, linestyle='--')
    k = 0
    fw = open(
        res_dir + 'speed_res/' + str(test_start) + 'day_' + time_period +
        '_roads_speed_compare.csv', 'w')
    fw.write(',' + ','.join(un_seeds) + '\n')
    for method in methods:
        # if method == 'ori':
        #     continue

        speed_arr = predict_res[method] * 100
        fw.write(method + ',')
        fw.write(','.join([str(round(speed, 1))
                           for speed in speed_arr]) + '\n')
        # if method == 'LSTM':
        #     method = 'OHKG'
        #     speed_arr += 4
        ax.plot(
            range(len(un_seeds)),
            speed_arr,
            label='$' + method + '$',
            marker=markers[k],
            linestyle='-',
            color=colors[k])
        k += 1

    fw.close()
    plt.legend(loc='upper right')
    plt.savefig(res_dir + 'speed_res/' + str(test_start) + 'day_' +
                time_period + '_methods_roads_speed.png')
    plt.close()
    return


def periods_compare(interval, train_end, test_start, params):

    periods_result = {}
    periods = 24 * 60 // interval
    methods = []
    for period in range(periods):
        params['time_period'] = str(period)
        time_s = time.time()
        predict_res, un_seeds = compare(interval, train_end, test_start,
                                        params)
        time_e = time.time()
        print(period, str(time_e - time_s) + 's')
        for method in predict_res:
            if method not in periods_result:
                methods.append(method)
                periods_result[method] = []
            periods_result[method].append(predict_res[method])

    ori_df = pd.DataFrame(
        periods_result['ori'], index=range(periods), columns=un_seeds)
    ori_df.to_csv(result_dir + test_start + 'day_ori_predict.csv', sep=',')
    mre_result = {}
    rmse_result = {}
    mae_result = {}
    for method in methods:
        if method == 'ori':
            continue
        df = pd.DataFrame(
            periods_result[method], index=range(periods), columns=un_seeds)
        df.to_csv(
            result_dir + test_start + 'day_' + method + '_predict.csv',
            sep=',')
        mae_result[method] = abs(df - ori_df)
        mre_result[method] = mae_result[method] / ori_df
        rmse_result[method] = (df - ori_df)**2

    markers = ['o', '*', 'D', '^']
    colors = ['r', 'b', 'y', 'g']

    ax = plt.subplot()
    ax.set_xlabel("roads_order")
    ax.set_ylabel('MRE')
    ax.yaxis.grid(True, linestyle='--')
    ax.xaxis.grid(True, linestyle='--')
    k = 0
    fw = open(result_dir + 'roads_mre.csv', 'w')
    fw.write(',' + ','.join(un_seeds) + '\n')
    for method in methods:
        if method == 'ori':
            continue
        fw.write(method + ',')
        mre_arr = np.mean(mre_result.values, axis=0)
        fw.write(','.join([str(round(v, 3)) for v in mre_arr]) + '\n')
        ax.plot(
            range(len(un_seeds)),
            mre_arr,
            label='$' + method + '$',
            marker=markers[k],
            linestyle='-',
            color=colors[k])
        k += 1

    fw.close()
    plt.legend(loc='best')
    plt.savefig(result_dir + 'roads_mre.png')
    plt.close()

    return


def dims_result(res_dir):
    speed_tensor = []
    un_seeds = []
    methods = ['ori', 'HAI', 'KNN', 'LSTM', 'ONE-HOP']
    periods = list(range(48))
    file_names = os.listdir(res_dir)
    file_names.sort(key=lambda x: int(x.split('_')[1]))
    for file in file_names:
        if file[-3:] != 'csv':
            continue

        df = pd.read_csv(res_dir + file, index_col=0)
        df = df.reindex(methods)
        if len(un_seeds) == 0:
            un_seeds = df.columns
            methods = df.index
        if methods.all() != df.index.all():
            print('err_file:', file)
            print(methods, df.index)
            break

        speed_tensor.append(df.values)

    speed_tensor = np.array(speed_tensor)

    diff_tensor = np.zeros((len(periods), 4, len(un_seeds)))

    for i in range(4):
        diff_tensor[:, i, :] = speed_tensor[:, i +
                                            1, :] - speed_tensor[:, 0, :]

    # YC = (abs(diff_tensor) / speed_tensor[:, 1:, :]) > 1
    # diff_tensor[YC] = np.mean(abs(diff_tensor))

    MAE_tensor = abs(diff_tensor)
    MRE_tensor = MAE_tensor / speed_tensor[:, 1:, :]
    RMSE_tensor = diff_tensor**2

    periods_res, roads_res = {}, {}
    eva_list = ['MAE', 'MRE', 'RMSE']
    periods_res['MAE'] = np.mean(MAE_tensor, axis=2)
    periods_res['MRE'] = np.mean(MRE_tensor, axis=2)
    periods_res['RMSE'] = np.sqrt(np.mean(RMSE_tensor, axis=2))

    roads_res['MAE'] = np.mean(MAE_tensor, axis=0)
    roads_res['MRE'] = np.mean(MRE_tensor, axis=0)
    roads_res['RMSE'] = np.sqrt(np.mean(RMSE_tensor, axis=0))

    # for eva in eva_list:
    #     scio.savemat(result_dir + 'periods_' + eva + '.mat',
    #                  {eva: periods_res[eva]})
    #     scio.savemat(result_dir + 'roads_' + eva + '.mat',
    #                  {eva: roads_res[eva]})

    markers = ['o', '*', 'D', '^']
    colors = ['r', 'b', 'y', 'g']

    for eva in eva_list:
        ax = plt.subplot()
        ax.set_xlabel("periods")
        ax.set_ylabel(eva)
        ax.yaxis.grid(True, linestyle='--')
        ax.xaxis.grid(True, linestyle='--')

        for i in range(4):
            ax.plot(
                periods,
                periods_res[eva][:, i],
                color=colors[i],
                marker=markers[i],
                label='$' + methods[i + 1] + '$',
                linestyle='-')

        plt.legend(loc='best')
        plt.savefig(result_dir + 'periods_compare_' + eva + '.png')
        plt.close()

    for eva in eva_list:
        ax = plt.subplot()
        ax.set_xlabel("roads_order")
        ax.set_ylabel(eva)
        ax.yaxis.grid(True, linestyle='--')
        ax.xaxis.grid(True, linestyle='--')

        for i in range(4):
            ax.plot(
                range(len(un_seeds)),
                roads_res[eva][i, :],
                color=colors[i],
                marker=markers[i],
                label='$' + methods[i + 1] + '$',
                linestyle='-')

        plt.legend(loc='best')
        plt.savefig(result_dir + 'roads_compare_' + eva + '.png')
        plt.close()

    return


def roads_plot(time_period):
    methods = ['HA', 'KNN', 'LSTM', 'ONE-HOP']
    eva_list = ['MAE', 'MRE', 'RMSE']
    markers = ['o', '*', 'D', '^']
    colors = ['r', 'b', 'y', 'g']

    for eva in eva_list:
        arr = scio.loadmat(result_dir + 'roads_' + eva + '.mat')[eva]
        print(arr.shape)
        # un_seeds_list = range(len(data))
        ax = plt.subplot()
        ax.set_xlabel("roads_order")
        ax.set_ylabel(eva)
        ax.yaxis.grid(True, linestyle='--')
        ax.xaxis.grid(True, linestyle='--')

        for i in range(4):
            data = arr[i, :]
            ax.plot(
                range(len(data)),
                data,
                color=colors[i],
                marker=markers[i],
                label='$' + methods[i] + '$',
                linestyle='-')

        plt.legend(loc='upper right')
        plt.savefig(result_dir + 'methods_compare/' + time_period + '_roads_' +
                    eva + '.png')
        plt.close()
    return


def periods_plot():
    methods = ['HA', 'KNN', 'LSTM', 'ONE-HOP']
    eva_list = ['MAE', 'MRE', 'RMSE']
    markers = ['o', '*', 'D', '^']
    colors = ['r', 'b', 'y', 'g']

    for eva in eva_list:
        arr = scio.loadmat(result_dir + 'periods_' + eva + '.mat')[eva]

        # un_seeds_list = range(len(data))
        ax = plt.subplot()
        ax.set_xlabel("periods")
        ax.set_ylabel(eva)
        ax.yaxis.grid(True, linestyle='--')
        ax.xaxis.grid(True, linestyle='--')

        for i in range(4):
            data = arr[:, i]
            ax.plot(
                range(len(data)),
                data,
                color=colors[i],
                marker=markers[i],
                label='$' + methods[i] + '$',
                linestyle='-')

        plt.legend(loc='upper right')
        plt.savefig(result_dir + 'methods_compare/' + time_period +
                    '_periods_' + eva + '.png')
        plt.close()
    return


if __name__ == '__main__':
    train_end = 7
    test_start = 8
    interval = 30

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

    dims_res_dir = result_dir + 'methods_compare/speed_res/'
    # dims_result(dims_res_dir)
    # for period in range(4,48):
    #     params['time_period'] = str(period)
    roads_result(interval, train_end, test_start, params)
    # roads_plot(time_period)
    # periods_plot()
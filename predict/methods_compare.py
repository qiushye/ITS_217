from init import dates, cur_dir, data_dir, result_dir
import numpy as np
import pandas as pd
import road_network
from sklearn.neighbors import KNeighborsRegressor
from lstm_model import lstm_predict
from predict import estimate
import os
import matplotlib.pyplot as plt


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
        arr = speed_tensor[:, :, i].T / 100
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
    test_date = params['test_date']
    sup_rate = params['sup_rate']

    roads_path = data_dir + 'road_map.txt'
    RN = road_network.roadmap(roads_path,
                              data_dir + str(interval) + '_impute/')
    for r in RN.roads:
        RN.get_info(r, data_dir, time_period, train_rate)
    RN.seed_select(seed_rate, train_rate, sup_rate)

    un_seeds = list(RN.roads.keys() - RN.seeds)
    un_seeds.sort()

    predict_result = {}

    # 真实值
    ori_Y_result = []
    for road in un_seeds:
        arr = RN.road_info[road].V[time_period][dates[test_start]]
        ori_Y_result.append(arr)

    ori_Y_result = np.array(ori_Y_result)
    predict_result['ori'] = ori_Y_result

    # knn预测
    knn_X_train, knn_Y_train, knn_X_test, knn_Y_test = data_knn(
        RN, interval, time_period, train_end, test_start)
    uni_knr = KNeighborsRegressor(weights='uniform')  #初始化平均回归的KNN回归器
    uni_knr.fit(knn_X_train, knn_Y_train)
    knn_Y_predict = uni_knr.predict(knn_X_test)
    knn_Y_predict = np.reshape(knn_Y_predict, len(un_seeds))
    predict_result['KNN'] = knn_Y_predict

    # lstm预测
    n_step = 3
    n_input = 1
    lstm_Y_predict = []
    for road in un_seeds:
        lstm_X_train, lstm_Y_train, lstm_X_test, lstm_Y_test = data_lstm(
            interval, time_period, road, n_step, train_end, test_start)
        prediction = lstm_predict(n_step, n_input, lstm_X_train, lstm_Y_train,
                                  lstm_X_test)
        lstm_Y_predict.append(prediction)

    lstm_Y_predict = np.array(lstm_Y_predict)
    lstm_Y_predict = lstm_Y_predict.reshape(len(un_seeds))
    predict_result['LSTM'] = lstm_Y_predict

    # one_hop_model预测
    est_RN, run_time, _ = estimate(RN, params)
    one_hop_Y_predict = []
    for road in un_seeds:
        arr = est_RN.road_info[road].V[time_period][dates[test_start]]
        one_hop_Y_predict.append(arr)

    one_hop_Y_predict = np.array(one_hop_Y_predict)
    one_hop_Y_predict = one_hop_Y_predict.reshape(len(un_seeds))
    predict_result['ONE-HOP'] = one_hop_Y_predict

    # 历史平均预测
    HAI_Y_predict = []
    for road in un_seeds:
        arr = np.mean(est_RN.road_info[road].V[time_period][:train_end])
        HAI_Y_predict.append(arr)

    HAI_Y_predict = np.array(HAI_Y_predict)
    HAI_Y_predict = HAI_Y_predict.reshape(len(un_seeds))
    predict_result['HAI'] = HAI_Y_predict

    return predict_result, un_seeds


def roads_result(interval, train_end, test_start, params):
    time_period = params['time_period']
    marker = ['o', '*', '_', '^']

    predict_res, un_seeds = compare(interval, train_end, test_start, params)
    ax = plt.subplot()
    ax.set_xlabel("roads_order")
    ax.set_ylabel('MRE')
    ax.yaxis.grid(True, linestyle='--')
    ax.xaxis.grid(True, linestyle='--')
    k = 0
    for method in predict_res:
        if method == 'ori':
            continue

        mre_arr = abs(predict_res[method] -
                      predict_res['ori']) / predict_res['ori']
        print(method, mre_arr.shape)
        ax.plot(
            range(len(un_seeds)),
            mre_arr,
            label='$' + method + '$',
            linestyle=marker[k],
            color='red')
        k += 1
    plt.legend(loc='best')
    plt.savefig(result_dir + time_period + '_methods_roads_MRE.png')
    plt.close()
    return


if __name__ == '__main__':
    train_end = 12
    test_start = 12
    interval = 30

    train_rate = (train_end - 1) / len(dates)
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
        'seed_rate': seed_rate
    }

    roads_result(interval, train_end, test_start, params)
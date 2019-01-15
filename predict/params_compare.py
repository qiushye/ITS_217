'''
一阶模型参数对结果的影响，
参数包括种子比例和训练天数
'''
from init import dates, data_dir, result_dir, weekday
dates = weekday
from predict import estimate, evaluate, predict, speed_refer, ga_knn_opt
import copy
import road_network
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def seed_rate_vary(RN, params):
    time_period = params['time_period']
    test_start = params['test_start']

    date_cols = list(range(test_start, len(dates)))
    print(dates[date_cols])
    rates = np.array([i * 10 for i in range(1, 7)])  # 种子数百分比
    rmse_df = pd.DataFrame(
        np.zeros((len(rates), len(date_cols))),
        index=rates,
        columns=dates[date_cols])
    mre_df = copy.deepcopy(rmse_df)
    for i in rates:
        params['seed_rate'] = i / 100

        for test_start in date_cols:
            temp_RN = copy.deepcopy(RN)
            test_date = dates[test_start]
            params['test_date'] = test_date
            est_RN, _, _ = ga_knn_opt(temp_RN, params)
            rmse, mre = evaluate(RN, est_RN, time_period, test_date)
            rmse_df[test_date][i] = rmse
            mre_df[test_date][i] = mre
    # print(rmse_df)

    res_dir = result_dir + 'params_compare/'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    path_prefix = res_dir + time_period + '_seed_rate_'

    markers = ['o', '*', 'D', '^']
    colors = ['b', 'r', 'y', 'g']
    style_dict = {}
    for col in date_cols:
        m = col // len(colors)
        c = col % len(colors)
        style_dict[dates[col]] = markers[m] + colors[c] + '--'

    rmse_df.to_csv(path_prefix + 'rmse.csv', columns=dates[date_cols], sep=',')
    mre_df.to_csv(path_prefix + 'mre.csv', columns=dates[date_cols], sep=',')

    rmse_df.plot(style=style_dict, grid=True)
    plt.xlabel("seeds_rate(%)")
    plt.ylabel('RMSE(100km/h)')
    plt.legend(loc='upper right')
    plt.savefig(path_prefix + 'rmse.png')
    plt.close()

    mre_df.plot(style=style_dict, grid=True)
    plt.xlabel("seeds_rate(%)")
    plt.ylabel('MRE')
    plt.legend(loc='upper right')
    plt.savefig(path_prefix + 'mre.png')
    plt.close()

    return


def train_end_vary(RN, params):
    time_period = params['time_period']

    cols = ['next-day', 'fixed-date']
    indexes = range(6, 10)
    rmse_df = pd.DataFrame(
        np.zeros((len(indexes), len(cols))), index=indexes, columns=cols)
    mre_df = copy.deepcopy(rmse_df)

    for train_end in indexes:
        # 训练集后一天的预测
        test_start = train_end
        train_rate = train_end / len(dates)
        test_date = dates[test_start]
        params['train_end'] = train_end
        params['test_start'] = test_start
        params['train_rate'] = train_rate
        params['test_date'] = test_date

        RN = road_network.roadmap(roads_path, train_end,
                                  data_dir + str(interval) + '_impute/')

        for r in RN.roads:
            RN.get_info(r, data_dir, time_period, train_rate)

        temp_RN = copy.deepcopy(RN)
        fixed_RN = copy.deepcopy(RN)

        est_RN, _, _ = estimate(temp_RN, params)
        rmse, mre = evaluate(RN, est_RN, time_period, test_date)
        rmse_df['next-day'][train_end] = rmse
        mre_df['next-day'][train_end] = mre

        # 固定日期的预测
        test_start = indexes[-1]
        fixed_date = dates[test_start]
        params['test_start'] = test_start
        params['test_date'] = fixed_date

        est_RN, _, _ = ga_knn_opt(fixed_RN, params)
        rmse, mre = evaluate(RN, est_RN, time_period, fixed_date)
        rmse_df['fixed-date'][train_end] = rmse
        mre_df['fixed-date'][train_end] = mre

    res_dir = result_dir + 'params_compare/'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    path_prefix = res_dir + time_period + '_train_'

    style_dict = {'next-day': 'ob-', 'fixed-date': 'or-'}
    # markers = ['o', 'o', '*', 'D', '^']
    # colors = ['b', 'r', 'b', 'y', 'g']

    rmse_df.to_csv(path_prefix + 'rmse.csv', columns=cols, sep=',')
    mre_df.to_csv(path_prefix + 'mre.csv', columns=cols, sep=',')

    rmse_df.plot(style=style_dict, grid=True)
    plt.xlabel("train days")
    plt.ylabel('RMSE(100km/h)')
    plt.legend(loc='upper right')
    plt.savefig(path_prefix + 'rmse.png')
    plt.close()

    mre_df.plot(style=style_dict, grid=True)
    plt.xlabel("train days")
    plt.ylabel('MRE')
    plt.legend(loc='upper right')
    plt.savefig(path_prefix + 'mre.png')
    plt.close()
    '''
    rmse_fw = open(res_dir + time_period + '_train_rmse.csv', 'w')
    rmse_fw.write(',' + ','.join([str(i) for i in indexes]) + '\n')
    k = 0
    for lb in column:
        rmse_list = rmse_df[lb]
        rmse_fw.write(lb + ',')
        rmse_fw.write(','.join([str(rmse) for rmse in rmse_list]) + '\n')

        ax = plt.subplot()
        ax.set_xlabel("train days")
        ax.set_ylabel('RMSE(100km/h)')
        ax.yaxis.grid(True, linestyle='--')
        ax.xaxis.grid(True, linestyle='--')

        ax.plot(
            indexes,
            rmse_list,
            label='$' + lb + '$',
            marker=markers[k],
            linestyle='-',
            color=colors[k])
        k += 1

    plt.legend(loc='upper right')
    plt.savefig(res_dir + time_period + '_train_rmse.png')
    plt.close()
    rmse_fw.close()

    mre_fw = open(res_dir + time_period + '_train_mre.csv', 'w')
    mre_fw.write(',' + ','.join([str(i) for i in indexes]) + '\n')
    k = 0
    for lb in column:
        mre_list = mre_df[lb]
        mre_fw.write(lb + ',')
        mre_fw.write(','.join([str(mre) for mre in mre_list]) + '\n')

        ax = plt.subplot()
        ax.set_xlabel("train days")
        ax.set_ylabel('MRE')
        ax.yaxis.grid(True, linestyle='--')
        ax.xaxis.grid(True, linestyle='--')

        ax.plot(
            indexes,
            mre_list,
            label='$' + lb + '$',
            marker=markers[k],
            linestyle='-',
            color=colors[k])
        k += 1

    plt.legend(loc='upper right')
    plt.savefig(res_dir + time_period + '_train_mre.png')
    plt.close()
    mre_fw.close()
    '''
    return


if __name__ == '__main__':
    raw_dir = 'D:/启东数据/启东流量数据/'
    roads_path = data_dir + 'road_map.txt'
    interval = 30
    train_end = 8
    test_start = 8

    # train_rate = 0.6
    train_rate = train_end / len(dates)
    time_period = '30'
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

    # seed_rate_vary(RN, params)
    train_end_vary(RN, params)

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from init import data_dir, dates, result_dir
import road_network
import os
import sys


def performance_metric(Y_true, Y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """

    score = r2_score(Y_true, Y_predict)

    return score


def fit_model_shuffle(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

    # Create a KNN regressor object
    regressor = KNeighborsRegressor()
    # Create a dictionary for the parameter 'n_neighbors' with a range from 3 to 10
    params = {'n_neighbors': range(3, 10)}

    # Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(
        regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


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


'''
interval = 30
times_data, roads = extract_data(data_dir, interval)
print(data_dir)
roads_path = data_dir + 'road_map.txt'
interval = 30
time_period = '35'
train_rate = 0.8
seed_rate = 0.3
sup_rate = 1
train_num = int(len(dates) * train_rate) + 1

RN = road_network.roadmap(roads_path, data_dir + str(interval) + '_impute/')
for r in roads:
    RN.get_info(r, data_dir, time_period, train_rate)
RN.seed_select(seed_rate, train_rate, sup_rate)

seeds = list(RN.seeds)
un_seeds = list(set(roads) - RN.seeds)

X_train = times_data[time_period].ix[:train_num, seeds].values
Y_train = times_data[time_period].ix[:train_num, un_seeds, ].values
X_test = times_data[time_period].ix[train_num:train_num + 1, seeds].values
Y_test = times_data[time_period].ix[train_num:train_num + 1, un_seeds].values
print(X_train.shape, X_test.shape)

uni_knr = KNeighborsRegressor(weights='uniform')  #初始化平均回归的KNN回归器
uni_knr.fit(X_train, Y_train)
Y_predict = uni_knr.predict(X_test)
print(Y_predict.shape)

print('R-squared value of uniform-weighted KNeighborsRegression:',
      uni_knr.score(X_test, Y_test))
print('The mean squared error of uniform-weighted KNeighborsRegression:',
      mean_squared_error(Y_test, Y_predict))
print('The mean absolute error of uniform-weighted KNeighborsRegression:', \
mean_absolute_error(Y_test, Y_predict))
print(np.mean(abs(Y_test - Y_predict) / Y_test))
'''
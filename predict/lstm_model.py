# Exampleof LSTM to learn a sequence
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from pre_process import speed_extract, data_dir
import os
import pandas as pd
import sys

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def split_dataset(dataset, train_rate, time_step):  # 生成训练集和测试集
    days = dataset.shape[0]
    X_train, X_test = [], []
    Y_train, Y_test = [], []
    for i in range(0, days - time_step):
        if (i + time_step) / days > train_rate:
            X_test = dataset[i + time_step - 1:i + time_step + time_step - 1]
            Y_test = dataset[i + time_step + time_step - 1]
            break
        X_train.append(dataset[i:i + time_step])
        Y_train.append(dataset[i + time_step])
        #print ("x ",i," ",i+time_step)
        #print ("y ",i+time_step," ",i+time_step+1)
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(
        Y_test)


def lstm_predict(n_step, n_input, X_train, Y_train, X_test):
    # 1. definenetwork
    model = Sequential()
    model.add(LSTM(len(X_train), input_shape=(n_step, n_input)))
    model.add(Dense(n_input))
    # 2.compile network
    model.compile(optimizer='adam', loss='mean_squared_error')
    # 3. fit network
    model.fit(
        X_train, Y_train, epochs=1000, batch_size=len(X_train), verbose=0)
    # 5. make predictions
    predictions = model.predict(X_test, verbose=0)
    # print(predictions, Y_test)
    # MRE = np.mean(abs(predictions - Y_test) / Y_test)
    return predictions


'''
# create sequence
interval = 30
id = '20'
time_period = '15'
data_dir += str(interval) + '_impute/'
df = pd.read_csv(data_dir + id + '.csv', index_col=0)
arr = df[time_period].values

train_rate = 0.6
days = arr.shape[0]

n_step = 4
n_input = 1
values = arr / 100
X_train, Y_train, X_test, Y_test = split_dataset(values, train_rate, n_step)
print(X_train.shape, Y_train.shape)
X_train = X_train.reshape(X_train.shape[0], n_step, n_input)
batch_size = len(X_train)
# Y_train = Y_train[:, 0, :]
X_test = X_test.reshape(1, n_step, n_input)

predictions = lstm_predict(n_step, n_input, X_train, Y_train, X_test)
MRE = np.mean(abs(predictions - Y_test) / Y_test)
print(MRE)
print(predictions.shape)
# sys.exit()

# 4. evaluate network
# loss = model.evaluate(X_test, Y_test, verbose=0)
'''
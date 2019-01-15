import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import sys
from init import result_dir

methods = ['KNN', 'LSTM', 'ONE-HOP', 'HA']
eva_list = ['MAE', 'MRE', 'RMSE']
markers = ['o', '*', 'D', '^']
colors = {'KNN': 'ro-', 'LSTM': 'b*--', 'ONE-HOP': 'y', 'HA': 'b'}
roads = range(34)

res_dict = {}
for m in methods:
    res_dict[m] = np.random.rand(6)

df = pd.DataFrame(
    np.array(list(res_dict.values())).T,
    columns=res_dict.keys(),
    index=list(range(5, 11)))
df = abs(df.sub(df['HA'], axis=0).div(df['HA'], axis=0))
df = df.drop(['HA'], axis=1)
df.to_csv('test.csv', sep=',')
df.plot(style=colors, grid=True)
plt.xlabel('times')
plt.ylabel('mre')
plt.show()

# df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
# ab = df.plot.scatter(x='a', y='b', color='DarkBlue', label='Group 1')
# abcd = df.plot.scatter(x='c', y='d', color='DarkGreen', label='Group 2', ax=ab)
# plt.show()
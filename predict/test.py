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

df = pd.DataFrame(np.random.rand(6, 4), columns=methods)
df.plot(style=colors)
plt.show()

# df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
# ab = df.plot.scatter(x='a', y='b', color='DarkBlue', label='Group 1')
# abcd = df.plot.scatter(x='c', y='d', color='DarkGreen', label='Group 2', ax=ab)
# plt.show()
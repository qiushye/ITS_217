"""
created by qiushye on 2018.11.7
python version >= 3
"""

import pandas as pd
import numpy as np


class road:
    def __init__(self, data_path):
        # data_path格式：id.csv, 行表示日期速度， 列表示时段速度
        self.id = data_path.split('.')[0]  # id为string格式
        self.start_id = None
        self.end_id = None
        self.max_v = 100
        df = pd.read_csv(data_path, index_col=0)
        for col in df.columns:
            for i in df.index:
                df[col][i] /= self.max_v  # 归一化
        self.V = df
        delta_V = df.copy()
        V_diff = df.copy()
        for l in df.columns:
            mean = np.mean(df[l])

            def vary(v1):
                if v1 >= mean:
                    return 1
                else:
                    return -1

            delta_V[l] = delta_V[l].map(vary)
            V_diff[l] = V_diff[l].map(lambda x: x - mean)

        self.delta_V = delta_V
        self.V_diff = V_diff
        self.A1 = set()  # 1-hop邻居
        self.A2 = set()  # 2-hop邻居
        self.UN = set()
        self.UE = set()  # 所有连接的边
        self.W = dict()
        self.correlations = dict()

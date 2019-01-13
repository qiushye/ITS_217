# -*- coding: UTF-8 -*-
"""
created by qiushye on 2018.11.7
python version >= 3
"""

from road import road
import numpy as np
import math
from random import random
from init import dates
import copy


class roadmap:
    def __init__(self, roads_path, train_end, data_dir):
        # 路网文件格式：road_id, start_id-end_id
        roads = dict()
        start_ids = dict()
        end_ids = dict()

        with open(roads_path, 'r') as f:
            # 将所有路段的起止端点记录好，为后续建立一阶模型提供基础
            for line in f:
                row = line.strip().split(',')
                road_id = row[0]
                [start_id, end_id] = row[1].split('-')
                roads[road_id] = (start_id, end_id)

                if start_id not in start_ids:
                    start_ids[start_id] = []
                start_ids[start_id].append(road_id)

                if end_id not in end_ids:
                    end_ids[end_id] = []
                end_ids[end_id].append(road_id)

        self.roads = roads
        self.road_info = dict()  # 路段的模型信息
        self.start_ids = start_ids  # 所有起始点的索引路段
        self.end_ids = end_ids  # 所有终止点的索引路段
        self.seeds = set()  # 路网的种子集合
        self.effect_rate = 0.9  # 传播因子
        self.max_level = 100  # 最大估计等级
        self.est_levels = dict()  # 估计等级索引
        self.known = dict()  # 已知性索引
        for r in roads:
            self.est_levels[r] = self.max_level
            self.known[r] = False

        for r in roads:  # 初始化路段模型数据
            rs = road(data_dir + r + '.csv', train_end)
            self.road_info[r] = rs

        # self.data = dict()
        # for r in roads:
        #     rs = road(r + '.csv')
        #     self.data[r] = rs

    def corr(self, id1, id2, time_period, rate):  # 相关性系数计算
        rs1 = self.road_info[id1]
        deltav1_list = rs1.delta_V[time_period]
        rs2 = self.road_info[id2]
        deltav2_list = rs2.delta_V[time_period]
        indexes = dates
        indice = 0
        same_num = 0
        for i in range(len(indexes)):
            if i / len(indexes) > rate:
                break
            indice += 1
            if deltav1_list[indexes[i]] == deltav2_list[indexes[i]]:
                same_num += 1
        return same_num / indice

    def get_1hop(self, id):  # 建立模型，只考虑上游影响
        start_id, end_id = self.roads[id]
        start_ids = self.start_ids
        end_ids = self.end_ids
        A1 = set()
        E1 = set()
        """
        try:
            for r in start_ids[end_id]:
                # 邻居路段第一类：以参考路段的终点作为起点
                if r == id:
                    continue
                if self.corr(id, r, time_period, rate) > corr_thre:
                    A1.add(r)
                    E1.add(id + '-' + r)
        except:
            pass
        """

        try:

            for r in end_ids[start_id]:
                # 邻居路段第二类：以参考路段的起点作为终点
                if end_id in start_ids and r in start_ids[end_id]:  # 去掉相向路段
                    continue
                edge = r + '-' + id
                A1.add(r)
                E1.add(edge)
        except:
            pass

        return A1, E1

    def get_info(self, id, data_dir, time_period, rate):  # 更新模型的信息
        rs = self.road_info[id]
        start_id, end_id = self.roads[id]
        rs.start_id = start_id
        rs.end_id = end_id

        A1, E1 = self.get_1hop(id)
        # print(id, A1)
        rs.A1 = A1
        rs.UE |= E1
        rs.UN |= A1
        # for r in rs.A1:
        #     A, E = self.get_1hop(r, time_period, rate, corr_thre)
        #     rs.A2 |= A
        #     rs.UE |= E

        # rs.UN |= rs.A2
        # rs.A2 -= rs.A1
        rs.UN.add(id)

        for e in rs.UE:
            r1, r2 = e.split('-')
            # rs.W[e] = self.corr(r1, r2, time_period, rate)
            rs.W[e] = random()
            rs.correlations[e] = self.corr(r1, r2, time_period, rate)

        self.road_info[id] = rs

        return

    def cov_sup(self, seed, sup_rate):  # 对于某个种子集合的支持数和覆盖数的加权和
        res = 0
        cov_set = set()

        for s in seed:
            for r in self.roads:
                if r in seed:
                    continue
                rs = self.road_info[r]
                if s in rs.UN:
                    cov_set.add(r)

        res += len(cov_set)
        if len(cov_set & seed) > 0:
            print('error')
        for cs in cov_set:
            rs = self.road_info[cs]

            res += len(rs.UN & seed) * sup_rate

        return res

    def seed_select(self, seed_rate, sup_rate):  # 贪心算法求种子集合
        seed = set()
        K = int(seed_rate * len(self.roads))
        next_seed = ''
        while len(seed) <= K:
            max_rise = 0
            cov_sup_pre = self.cov_sup(seed, sup_rate)
            for r in self.roads:
                if r in seed:
                    continue

                cov_sup_next = self.cov_sup(seed | set([r]), sup_rate)
                cur_rise = cov_sup_next - cov_sup_pre
                # print(cov_sup_next, cov_sup_pre)

                if cur_rise > max_rise:
                    max_rise = cur_rise
                    next_seed = r
                    # print(r, cur_rise)

            un_seeds = self.roads.keys() - seed
            if next_seed not in seed:
                seed.add(next_seed)
            else:
                seed.add(un_seeds.pop())
            # break
        self.seeds = seed
        return seed

    def speed_diff_est(self, id, date, time_period):  # 速度差值估计
        rs = self.road_info[id]
        # UnS = rs.UN & self.seeds
        sde = 0
        for r in rs.A1:
            edge = r + '-' + id
            temp_rs = self.road_info[r]
            # temp_sde = 0
            # level = self.est_levels[r]

            temp_sde = temp_rs.V_diff[time_period][date] * self.effect_rate

            sde += temp_sde * rs.W[edge] * rs.correlations[edge]

        return sde

    def weight_learn(self, id, rate, time_period, threshold, alpha):  # 权值学习
        # seed = self.seeds
        rs = self.road_info[id]
        indexes = dates
        delta_fun = dict()
        for edge in rs.UE:
            delta_fun[edge] = 0
        iter = 0
        while True:

            indice = 0
            diff_pre = []
            while indice / len(indexes) < rate:
                date = indexes[indice]
                diff_pre.append(self.speed_diff_est(id, date, time_period))
                indice += 1
            # mean_pre = sum(diff_pre) / indice
            delta_fun_pre = copy.deepcopy(delta_fun)
            for edge in rs.W:
                temp1 = 0
                if id in edge:
                    other_road = edge.split('-')[0]
                    other_rs = self.road_info[other_road]
                    # break
                    for i in range(indice):
                        date = indexes[i]
                        v_diff = rs.V_diff[time_period][date]

                        other_v_diff = other_rs.V_diff[time_period][date]
                        temp1 += (diff_pre[i] - v_diff) * other_v_diff \
                        * self.effect_rate * rs.correlations[
                            edge]
                    # print(edge, temp1)
                    delta_fun[edge] = alpha * temp1 / indice
                    rs.W[edge] = max(rs.W[edge] - alpha * temp1 / indice, 0)
                    rs.W[edge] = min(rs.W[edge], 1)

            self.road_info[id] = rs
            rs = self.road_info[id]

            diff = []
            for i in range(indice):
                date = indexes[i]
                diff.append(self.speed_diff_est(id, date, time_period))
            # mean = sum(diff) / indice
            delta_diff = 0
            for edge in rs.UE:
                delta_diff += abs(delta_fun[edge] - delta_fun_pre[edge])
            if delta_diff / len(rs.UE) < threshold:  # 收敛条件为相邻迭代的梯度增长值小于阈值
                break

            iter += 1
            # break
        self.road_info[id] = rs
        return

    def online_est(self, id, date, time_period, rate):  # 在线估计
        rs = self.road_info[id]
        indexes = rs.V.index
        indice = 0
        while indice / len(indexes) < rate:
            indice += 1
        v_mean = np.mean(rs.V[time_period].values[:indice])
        if len(rs.UN & self.seeds) == 0:
            return v_mean

        v_diff_est = self.speed_diff_est(id, date, time_period)
        v_est = v_mean + v_diff_est

        return v_est

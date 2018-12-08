"""
created by qiushye on 2018.11.7
python version >= 3
"""

from road import road
import numpy as np
import math
from random import random
import copy


class roadmap:
    def __init__(self, roads_path, data_dir):
        # 路网文件格式：road_id, start_id-end_id
        roads = dict()
        start_ids = dict()
        end_ids = dict()

        with open(roads_path, 'r') as f:
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
        self.road_info = dict()
        self.start_ids = start_ids
        self.end_ids = end_ids
        self.seeds = set()
        self.effect_rate = 0.9
        self.max_level = 100
        self.est_levels = dict()
        self.known = dict()
        for r in roads:
            self.est_levels[r] = self.max_level
            self.known[r] = False

        for r in roads:
            rs = road(data_dir + r + '.csv')
            self.road_info[r] = rs

        # self.data = dict()
        # for r in roads:
        #     rs = road(r + '.csv')
        #     self.data[r] = rs

    def corr(self, id1, id2, time_period, rate):
        rs1 = self.road_info[id1]
        deltav1_list = rs1.delta_V[time_period]
        rs2 = self.road_info[id2]
        deltav2_list = rs2.delta_V[time_period]
        indexes = rs1.V.index
        indice = 0
        same_num = 0
        for i in range(len(indexes)):
            if i / len(indexes) > rate:
                break
            indice += 1
            if deltav1_list[indexes[i]] == deltav2_list[indexes[i]]:
                same_num += 1
        return same_num / indice

    def get_1hop(self, id, time_period, rate):
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

    def get_info(self, id, data_dir, time_period, rate):
        rs = self.road_info[id]
        start_id, end_id = self.roads[id]
        rs.start_id = start_id
        rs.end_id = end_id

        A1, E1 = self.get_1hop(id, time_period, rate)
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

    def cov_sup(self, seed, sup_rate):
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

    def seed_select(self, K, time_period, rate, sup_rate):
        seed = set()

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
            seed.add(next_seed)
            # break
        self.seeds = seed
        return seed

    def trend_infer(self, id, date, time_period, rate):
        rs = self.road_info[id]
        seed = self.seeds
        p_max = 0
        delta_v_max = -1
        delta_v_dict = dict()
        for r in rs.UN:
            if r in seed:
                rs_temp = self.road_info[r]
                delta_v_dict[r] = rs_temp.delta_V[time_period][date]
            else:
                delta_v_dict[r] = -1
        non_seed = list(rs.UN - seed)

        for i in range(2**len(non_seed)):
            p = 0
            bin_arr = list(map(int, bin(i)[2:]))
            index_diff = len(non_seed) - len(bin_arr)
            for j in range(len(bin_arr)):
                if bin_arr[j] == 1:
                    r_id = non_seed[index_diff + j]
                    delta_v_dict[r_id] = 1

            for edge in rs.UE:
                sid, eid = edge.split('-')
                if delta_v_dict[sid] == delta_v_dict[eid]:
                    p += math.log10(self.corr(sid, eid, time_period, rate))
                else:
                    p += math.log10(
                        1 - min(self.corr(sid, eid, time_period, rate), 0.999))

            if p > p_max:
                p_max = p
                delta_v_max = delta_v_dict[rs.id]
        return delta_v_max

    def speed_diff_est(self, id, date, time_period):
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

    def weight_learn(self, id, rate, time_period, threshold, alpha):
        # seed = self.seeds
        rs = self.road_info[id]
        indexes = rs.V.index
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
            pre_diff_arr = np.array(diff_pre)
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
                        temp1 += (diff_pre[i] - v_diff) * other_v_diff
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
            diff_arr = np.array(diff)
            abs_diff = np.linalg.norm(diff_arr - pre_diff_arr)
            # print(diff_arr, pre_diff_arr)
            # print('diff_norm', abs_diff)
            # print(np.linalg.norm(pre_diff_arr))
            delta_diff = 0
            for edge in rs.UE:
                delta_diff += abs(delta_fun[edge] - delta_fun_pre[edge])
            if delta_diff / len(rs.UE) < threshold:
                break

            # if abs_diff < threshold:
            #     # if abs_diff < threshold:
            #     break
            iter += 1
            # break
        self.road_info[id] = rs
        # print('iter:', iter, delta_diff / len(rs.UE))
        # print(delta_fun, delta_fun_pre)
        return

    def online_est(self, id, date, time_period, rate):
        rs = self.road_info[id]
        indexes = rs.V.index
        indice = 0
        while indice / len(indexes) < rate:
            indice += 1
        v_mean = np.mean(rs.V[time_period].values[:indice])
        if len(rs.UN & self.seeds) == 0:
            return v_mean

        # delta_v = self.trend_infer(id, date, time_period, rate)
        v_diff_est = self.speed_diff_est(id, date, time_period)
        # print('diff_est', v_diff_est)
        # print('mean', v_mean)
        v_est = v_mean + v_diff_est

        return v_diff_est, v_est

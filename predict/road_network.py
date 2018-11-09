"""
created by qiushye on 2018.11.7
python version >= 3
"""

from road import road
import numpy as np
import math
from random import random


class roadmap:

    def __init__(self, roads_path):
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
        self.start_ids = start_ids
        self.end_ids = end_ids
        self.seeds = set()

        # self.data = dict()
        # for r in roads:
        #     rs = road(r + '.csv')
        #     self.data[r] = rs

    def get_1hop(self, id):
        start_id, end_id = self.roads[id]
        start_ids = self.start_ids
        end_ids = self.end_ids
        A1 = set()
        E1 = set()
        for r in start_ids[end_id]:
            # 邻居路段第一类：以参考路段的终点作为起点
            if r == id:
                continue
            A1.add(r)
            E1.add(id+'-'+r)

        for r in end_ids[start_id]:
            # 邻居路段第二类：以参考路段的起点作为终点
            if r == id:
                continue
            A1.add(r)
            E1.add(r+'-'+id)

        return A1, E1

    def get_info(self, id):
        rs = road(id+'.csv')
        start_id, end_id = self.roads[id]
        rs.start_id = start_id
        rs.end_id = end_id

        A1, E1 = self.get_1hop(id)
        rs.A1 = A1
        rs.UE |= E1
        for r in rs.A1:
            A, E = self.get_1hop(r)
            rs.A2 |= A
            rs.UE |= E

        rs.UN = rs.A2
        rs.UN.add(id)
        rs.A2 -= rs.A1

        rs.W = dict()
        for e in rs.UE:
            rs.W[e] = random()

        return rs

    def corr(self, id1, id2, period, rate):
        rs1 = road(id1 + '.csv')
        deltav1_list = rs1.delta_V[period]
        rs2 = road(id2 + '.csv')
        deltav2_list = rs2.delta_V[period]
        indexes = rs1.V.index
        count = 0
        same_num = 0
        for i in range(len(indexes)):
            if i / len(indexes) > rate:
                break
            count += 1
            if deltav1_list[indexes[i]] == deltav2_list[indexes[i]]:
                same_num += 1
        return same_num/count

    def trend_infer(self, id, date, time_period, seed, rate):
        rs = self.get_info(id)
        p_max = 0
        delta_v_max = -1
        delta_v_dict = dict()
        for r in rs.UN:
            if r in seed:
                rs_temp = road(r)
                delta_v_dict[r] = rs_temp.delta_V[time_period][date]
            else:
                delta_v_dict[r] = -1
        non_seed = list(rs.UN-seed)

        for i in range(2 ** len(non_seed)):
            p = 0
            bin_arr = list(map(int, bin(i)[2:]))
            index_diff = len(non_seed)-len(bin_arr)
            for j in range(len(bin_arr)):
                if bin_arr[j] == 1:
                    r_id = non_seed[index_diff + j]
                    delta_v_dict[r_id] = 1

            for edge in rs.UE:
                sid, eid = edge.split('-')
                if delta_v_dict[sid] == delta_v_dict[eid]:
                    p += math.log10(self.corr(sid, eid, time_period, rate))
                else:
                    p += math.log10(1 - self.corr(sid, eid, time_period, rate))

            if p > p_max:
                p_max = p
                delta_v_max = delta_v_dict[id]
        return delta_v_max

    def speed_diff_est(self, id, date, time_period):
        rs = road(id)
        UnS = rs.UN & self.seeds
        sde = 0
        for r in rs.A1:
            temp_rs = road(r)
            temp_sde = 0
            if r in self.seeds:
                temp_sde = temp_rs.V_diff[time_period][date]

            else:
                for s in UnS:
                    seed_temp_rs = road(s + '.csv')
                    temp_diff = seed_temp_rs.V_diff[time_period][date]
                    if s + '-' + r in rs.W in rs.W:
                        temp_sde += rs.W[s + '-' + r]*temp_diff
                    elif r + '-' + s in rs.W:
                        temp_sde += rs.W[r + '-' + s] * temp_diff

            if id + '-' + r in rs.W:
                sde += temp_sde * rs.W[id + '-' + r]
            else:
                sde += temp_sde * rs.W[r + '-' + id]

        return sde

    # 是否同时考虑上游和下游，整体优化可能会产生死循环

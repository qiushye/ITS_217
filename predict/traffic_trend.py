#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 18:31:56 2017

@author: qiushye
"""
import seed
import math

road_id_dir = "/home/qiushye/road_id_division_test2/"
threshold = 0.6
seed_rate = 0.15
alpha = 1
SEED = seed.seed_select(road_id_dir,threshold,seed_rate,alpha)
all_road_info = seed.all_road_info
c_weekday_d = seed.c_weekday_d
c_weekend_d = seed.c_weekend_d
period = seed.period
'''
fw = open("/home/qiushye/road_est.txt",'w+') 
for road_ in all_road_info.keys():
    fw.write(road_+'\n')
'''
def create_graph(road):
    if road in SEED:
        exit -1
    onehop_neighbor = all_road_info[road][-1]
    twohop_neighbor = set()
    #求2-hop邻居
    
    for road1 in onehop_neighbor:
        twohop_neighbor = twohop_neighbor|set(all_road_info[road1][-1])
    twohop_neighbor = twohop_neighbor-set([road])
    
    #road_neighbor = list(set(set(onehop_neighbor)|twohop_neighbor))
    Ue = []
    Un = []
    R1 = 0
    R2 = 0
    
    for rec in onehop_neighbor:
        if seed.get_road_R(rec,road,all_road_info,c_weekday_d,c_weekend_d,threshold) == (1,1):
            Un.append(rec)
            Ue.append([road,rec])
    A1 = Un
    Un.append(road)
    A2 = []
    for road1 in A1:
        temp_neighbor = all_road_info[road1][-1]
        for road2 in temp_neighbor:
            if road2 in SEED and road2 not in Un and seed.get_road_R(road1,road2,all_road_info,c_weekday_d,c_weekend_d,threshold) == (1,1):
                Un.append(road2)
                A2.append(road2)
                Ue.append([road1,road2])
    for set1 in A1:
        for set2 in A1:
            if set1 != set2:
                R1,R2 = seed.get_road_R(set1,set2,all_road_info,c_weekday_d,c_weekend_d,threshold)
                if R1 == 1 and R2 == 1 and [set2,set1] not in Ue:
                    Ue.append([set1,set2])
                
    return Un,Ue
#只考虑工作日
def potential(road1,road2,delta_v1,delta_v2):
    road1_info = all_road_info[road1]
    road2_info = all_road_info[road2]
    weekday_equal_rate = 0
    weekend_equal_rate = 0
    for date in c_weekday_d:
        if road1_info[1][date] == road2_info[1][date]:
            weekday_equal_rate += 1
    for date in c_weekend_d:
        if road1_info[3][date] == road2_info[3][date]:
            weekend_equal_rate += 1
    Cor =  float(weekday_equal_rate)/len(c_weekday_d) #+ float(weekend_equal_rate/len(c_weekend_d)))/2
    if delta_v1 == delta_v2:    
        f = Cor
    else:
        f = 1-Cor
    return f

def trend_infer(road,Un,Ue):
    #Un,Ue = create_graph(road)
    p_max = 0
    delta_v_max = -1
    p = 0
    delta_v = {}
    for road3 in list(set(Un)-set(SEED)):
        delta_v[road3] = [1,-1]
    for road1 in list(set(Un)-set(SEED)):
        for road2 in list(set(Un)-set(SEED)):
            #p = 0
            if [road1,road2] in Ue:
                for i in range(2):
                    for j in range(2):
                        f = potential(road1,road2,delta_v[road1][i],delta_v[road2][j])
                        
                        if f == 0:
                            p = p + math.log(1-threshold)
                        else:
                            p = p + math.log(f)
            if p > p_max:
                p_max = p
                delta_v_max = delta_v[road]
    return delta_v_max

#print create_graph('59567214784')
#print potential('59567202859','59567214784','2015-11-18')
#print trend_infer('59567202859','2015-11-18')

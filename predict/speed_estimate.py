#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 20:38:04 2017

@author: qiushye
"""

import traffic_trend
from random import random


SEED = traffic_trend.SEED
all_road_info = traffic_trend.all_road_info
c_weekday_d = traffic_trend.c_weekday_d
c_weekend_d = traffic_trend.c_weekend_d
all_road = list(all_road_info.keys())
theta = 0.05
lamda = 1
Tcon = 0.01
def get_weight(W,road,Un,Ue):
    
    #Un,Ue = traffic_trend.create_graph(road)
    #A = [val for val in Un if val != road]
    A = list(set(Un)&set(all_road_info[road][-1]))
    UnS = [val for val in Un if val in SEED]
    AS = [val for val in A if val in SEED]
    A_S  = [val for val in A if val not in SEED]
    #print A
    #print UnS
    if len(AS) == 0:
        return W
    
    elif len(AS) >= len(set(A_S)):
        Un = UnS
        A = AS
        #W = weight_learning(road,Un,Ue,W,theta,lamda,Tcon,v_diff_temp)
    else:
        for road1 in A_S:
            Un1,_ = traffic_trend.create_graph(road1)         #???bug
            #print Un1
            A1 = list(set(Un1)&set(all_road_info[road1][-1]))
            A1S = list(set(A1)&set(SEED))
            A1_S = list(set(A1)-set(SEED))
            if len(A1S) == 0 or len(A1S) < len(A1_S): #1表示road1本身
                Un = list(set(Un)-set([road1]))
                A = list(set(A)-set([road1]))
    A_S = [val for val in A if val not in SEED]
    UnS = [val for val in Un if val in SEED]
    for road1 in A:
        W[road1] = random()
    for road_j in A_S:
        for road_l in UnS:
            if [road_l,road_j] in Ue or [road_j,road_l] in Ue:
                #格式road1-road2
                W[road_l+'-'+road_j] = random()
    
    return W

#求UnS中的速度差异
def V_diff(road,date):
    Un,_ = traffic_trend.create_graph(road)
    
    UnS = [val for val in Un if val in SEED]
    v_diff = {}
    for road_l in UnS:
        if date in c_weekday_d:
            v_equal = all_road_info[road_l][4]
            v = all_road_info[road_l][0][date]
            v_diff[road_l] = abs(v-v_equal)
        if date in c_weekend_d:
            v_equal = all_road_info[road_l][5]
            v = all_road_info[road_l][2][date]
            v_diff[road_l] = abs(v-v_equal)
    return v_diff

def speed_diff(W,road,Un,Ue,v_diff):
    #1-hop相关邻居
    A = list(set(Un)&set(all_road_info[road][-1]))
    
    
    
    AS = [val for val in A if val in SEED]
    #W = get_weight(road)
    #v_diff = V_diff(road,date)
    diff_est = {}
    
    res = 0
    A_S = [val for val in A if val not in SEED]
    #W = get_weight({},road,Un,Ue)
    
    if len(AS) == 0:
        v_diff[road] = 0
    elif len(AS) >= len(set(A_S)):
        A = AS
        #W = weight_learning(road,Un,Ue,W,theta,lamda,Tcon,v_diff_temp)
    else:
        for road1 in A_S:
            Un1,_ = traffic_trend.create_graph(road1)         #???bug
            #print Un1
            A1 = list(set(Un1)&set(all_road_info[road1][-1]))
            A1S = list(set(A1)&set(SEED))
            A1_S = list(set(A1)-set(SEED))
            if len(A1S) == 0 or len(A1S) < len(A1_S): #1表示road1本身
                Un = list(set(Un)-set([road1]))
                A = list(set(A)-set([road1]))
                #W = weight_learning(road1,Un1,Ue1,W,theta,lamda,Tcon,v_diff_temp)
        
        #W = weight_learning(road,Un,Ue,W,theta,lamda,Tcon,v_diff_temp)  
    
    #UnS = [val for val in Un if val in SEED]
    #A = list(set(Un)&set(all_road_info[road][-1]))
    A_S = [val for val in A if val not in SEED]
    UnS = [val for val in Un if val in SEED]
    for road_j in A:
        if road_j in SEED:
            #print road_j
            diff_est[road_j] = v_diff[road_j]
            
        else:
            temp = 0
            for road_l in UnS:
                if road_l+'-'+road_j in W.keys():
                    temp += W[road_l+'-'+road_j]*v_diff[road_l]
            diff_est[road_j] = temp
    #print v_diff
    #print diff_est
    #print W
    #A会发生改变?
    for road_j in A:
        res += W[road_j]*diff_est[road_j]
    return res
#待修改
def weight_learning(road,Un,Ue,W,theta,lamda,Tcon,v_diff_):
    #W0 = get_weight(W,road,Un,Ue)
    
    V_diff_road = {}
    #V_diff_UnS = {}
    V_diff_est = {}
    #v_diff_final = {}
    # A = [val for val in Un if val != road]
    # UnS = [val for val in Un if val in SEED]
    
    for date in c_weekday_d:
            v_equal = all_road_info[road][4]
            v = all_road_info[road][0][date]
            V_diff_road[date] = abs(v-v_equal)
            #V_diff_UnS[road][date] = V_diff(road_uns,date)
    
    v_diff_sum = 0
    for date in c_weekday_d:
            #v_diff = V_diff(road,date)
            
            V_diff_est[date] = speed_diff(W,road,Un,Ue,v_diff_)
            v_diff_sum += V_diff_est[date]
    v_diff_new = v_diff_sum/len(c_weekday_d)
    i = 0
    while 1:
        v_diff_org = v_diff_new
        #theta = 0.09
        #lamda = 1
        #Tcon = 0.001
        for value in W:
            if '-' not in value:
                #v_diff_j = {}
            #if 1:
                sum_j = 0
                for date in c_weekday_d:
                    #v_equal_j = all_road_info[value][4]
                    #v_j = all_road_info[value][0][date]
                    if value in v_diff_.keys() and v_diff_[value] != 0 :
                        v_diff_j = v_diff_[value]
                    else:
                        Un_value,Ue_value = traffic_trend.create_graph(value)
                        '''
                        #W0 = get_weight(W0,value,Un_value,Ue_value)
                        W_value = weight_learning(value,Un_value,Ue_value,W0,theta,lamda,Tcon)
                        v_diff_value = V_diff(value,date)
                        v_diff_j = speed_diff(W_value,value,Un_value,Ue_value,v_diff_value)
                        v_diff_final[value][date] = v_diff_j'''
                        W_value = get_weight({},value,Un_value,Ue_value)
                        #嵌套循环?
                        #W_value = weight_learning(value,Un_value,Ue_value,W_value,theta,lamda,Tcon,v_diff_)
                        v_diff_j = speed_diff(W_value,value,Un_value,Ue_value,v_diff_)#abs(speed_est(W,v_diff_,theta,lamda,Tcon)-all_road_info[value][0][date])
                    sum_j += (V_diff_est[date]-V_diff_road[date])*v_diff_j
                W[value] -= theta*(sum_j/len(c_weekday_d)+lamda*W[value])
                if W[value] < 0:
                    W[value] = 0
                if W[value] > 1:
                    W[value] = 1
            '''for road_j in A:
                if [value,road_j] not in W.keys() or [road_j,value] not in W.keys():
                    continue'''
            if '-' in value:
                #print value
                #v_diff_j = {}
                sum_lj = 0
                road_l = value.split('-')[0]
                road_j = value.split('-')[1]
                
                for date in c_weekday_d:
                    v_equal_l = all_road_info[road_l][4]
                    v_l = all_road_info[road_l][0][date]
                    #Un_value,Ue_value = traffic_trend.create_graph(value)
                    v_diff_l = abs(v_l-v_equal_l)
                    sum_lj += (V_diff_est[date]-V_diff_road[date])*v_diff_l*W[road_j]   
                W[value] -= theta*(sum_lj/len(c_weekday_d)+lamda*W[value])
                if W[value] < 0:
                    W[value] = 0
                if W[value] > 1:
                    W[value] = 1
        v_diff_sum = 0
        for date in c_weekday_d:
            V_diff_est[date] = speed_diff(W,road,Un,Ue,v_diff_)
            v_diff_sum += V_diff_est[date]
        v_diff_new = v_diff_sum/len(c_weekday_d)
        #print v_diff_new
        i += 1
        if abs(v_diff_new-v_diff_org) < Tcon or i > 200:
            #print "-----"
            break
    return W

def choose_date(seed,period):
    weekday = ['1','2','3','4','5']
    common_weekday_date = []
    for road in seed:
        weekday_date_count = []
        with open(traffic_trend.road_id_dir+road+'.txt','r') as f:
            for line in f:
                record = line.strip().split(',')
                if record[8] in weekday and record[7] not in c_weekday_d:
                    if int(record[9]) - int(period) == -1:
                        temp_v = float(record[6])
                    if record[9] == period:
                        weekday_date_count.append(record[7])
                        temp_v = 0
                    if int(record[9]) - int(period) == 1 and temp_v != 0 and float(record[6]) - temp_v < 4:
                        weekday_date_count.append(record[7])
        if road == seed[0]:
            common_weekday_date = weekday_date_count
        common_weekday_date = [val for val in common_weekday_date if val in weekday_date_count]
    
    return common_weekday_date[0:4]

def speed_est(road,v_diff_temp,theta,lamda,Tcon,v_diff_):
    if road in SEED:
        return -1
    
    #global Un,Ue
    Un,Ue = traffic_trend.create_graph(road)
    
    UnS = [val for val in Un if val in SEED]
    A = list(set(Un)&set(all_road_info[road][-1]))
    AS = [val for val in A if val in SEED]
    v_equal = all_road_info[road][4]
    delta_v = traffic_trend.trend_infer(road,Un,Ue)
    
    A_S = [val for val in A if val not in SEED]
    #W = get_weight({},road,Un,Ue)
    #print W
    
    if len(AS) == 0:
        #v_diff_temp[road][date] = 0
        return v_equal
    elif len(AS) >= len(A_S):
        Un = UnS
        A = AS
        #W = weight_learning(road,Un,Ue,W,theta,lamda,Tcon,v_diff_temp)
    else:
        for road1 in A_S:
            
            Un1,_ = traffic_trend.create_graph(road1)         #???bug
            A1 = list(set(Un1)&set(all_road_info[road][-1]))
            #print Un1
            A1S = list(set(A1)&set(SEED))
            A1_S = list(set(A1)-set(SEED))
            if len(A1S) == 0 or len(A1S) < len(A1_S): #1表示road1本身
                Un = list(set(Un)-set([road1]))
                A = list(set(A)-set([road1]))
    
    W = get_weight({},road,Un,Ue)
    #print W
    if len(UnS) == 0:
        return v_equal
    W = weight_learning(road,Un,Ue,W,theta,lamda,Tcon,v_diff_)    
    
    
    v_diff = speed_diff(W,road,Un,Ue,v_diff_temp)
    v_diff_temp[road] = v_diff
    #v_diff = speed_diff(W,road,Un,Ue,v_diff_temp)
    #v_diff_temp[road] = v_diff
    #print v_diff
    if delta_v > 0:
        v_est = v_equal + v_diff
    else:
        v_est = v_equal - v_diff
    return v_est


'''
v_est_all = {}
for road in set(all_road)-set(SEED):
    v_est_all[road] = {}

for date in date_choose:
    v_diff_temp = V_diff() 
for road in set(all_road)-set(SEED):
    for date in date_choose:
        v_diff_temp = V_diff(road,date)
    v_est_all[road][date] = speed_est(road,date)
'''
date_choose = choose_date(SEED,traffic_trend.period)
v_est_all = {}
global v_diff_equal
v_diff_equal = {}
v_equal1 = {}
v_diff1 = {}
v_diff2 = {}
v_diff0 = {}
v_diff_temp = {}
v_temp1 = {}
v_diff_ori = {}
non_seed_road = [val for val in all_road if val not in SEED]
NSR_order = []
seed_num = {}
accu_rate ={}

for road in non_seed_road:
    #v_est_all[road] = 0
    v_temp1[road] = 0
    seed_num[road] = 0
    accu_rate[road] = 0
for road in SEED:
    v_diff1[road] = 0
    v_diff0[road] = 0
    #v_est_all[road] = 0
    v_diff_temp[road] = {}

for road1 in SEED:
    for date in c_weekday_d:
        v_diff_temp0 = abs(all_road_info[road1][0][date]-all_road_info[road1][4])
        v_diff0[road1] += v_diff_temp0
    v_diff_ori[road1] = v_diff0[road1]/len(c_weekday_d)

        
for road1 in SEED:
    for date in date_choose:
        with open(traffic_trend.road_id_dir+road1+'.txt','r') as f:
            for line in f:
                record = line.strip().split(',')
                if record[7] == date and record[9] == traffic_trend.period:
                    v_diff_temp[road1][date] = abs(float(record[6])-all_road_info[road1][4])
        v_diff1[road1] += v_diff_temp[road1][date]
for road1 in SEED:
    v_diff_equal[road1] = v_diff1[road1]/len(date_choose)
#非种子的实际速度
period = traffic_trend.period
for road1 in non_seed_road:
    date_num = 0
    for date in date_choose:
        with open(traffic_trend.road_id_dir+road1+'.txt','r') as f:
            v = 0
            for line in f:
                record = line.strip().split(',')
                if record[7] == date:
                    if int(record[9]) - int(period) == -1:
                        temp_v = float(record[6])
                    elif record[9] == period:
                        v = float(record[6])
                        temp_v = 0
                    elif int(record[9]) - int(period) == 1 and temp_v != 0 and float(record[6]) - temp_v < 4:
                        v = (temp_v + float(record[6]))/2 
                    v_temp1[road1] += v
                    if v != 0:
                        date_num += 1
    v_equal1[road1] = v_temp1[road1]/date_num
    
#记录非种子路段的Un中的种子个数
for road in non_seed_road:
    Un1,Ue1 = traffic_trend.create_graph(road)
    A1 = list(set(Un1)&set(all_road_info[road][-1]))
    A1S = [val for val in A1 if val in SEED]
    seed_num[road] = len(A1S)
'''
value = list(seed_num.values()).sort(reverse = True)
for i in value:
    index1 = seed_num.values().index(i)
    NSR_order.append(seed_num.keys()[index1])
'''
n= 0
#Un无种子
for road in non_seed_road:
    Un1,Ue1 = traffic_trend.create_graph(road)
    UnS1 = [val for val in Un1 if val in SEED]
    A1 = list(set(Un1)&set(all_road_info[road][-1]))
    A1S = [val for val in A1 if val in SEED]
    #print UnS1
    if len(A1S) == 0:
            v_est_all[road] = all_road_info[road][4]
            v_diff_equal[road] = 0
            n +=1
#Un的1-hop种子多于非种子
for road in non_seed_road:
        Un1,Ue1 = traffic_trend.create_graph(road)
        UnS1 = [val for val in Un1 if val in SEED]
        A1 = list(set(Un1)&set(all_road_info[road][-1]))
        A1S = [val for val in A1 if val in SEED]
        if len(A1S) > 0:
            v = speed_est(road,v_diff_equal,theta,lamda,Tcon,v_diff_ori)
            #W = get_weight({},road,Un1,Ue1)
            #W = weight_learning(road,Un1,Ue1,W,theta,lamda,Tcon,v_diff)
        
            #v = speed_est(road,v_diff_temp,theta,lamda,Tcon)
            v_est_all[road] = v
                
equal_num = 0                     
for road in non_seed_road:
    if v_est_all[road] == all_road_info[road][4]:
        equal_num += 1

accu_rate_sum = 0
for road in non_seed_road:
    accu_rate[road] = abs(v_est_all[road]-v_equal1[road])
    accu_rate_sum += accu_rate[road]/v_equal1[road]
mape = accu_rate_sum/(len(non_seed_road))

accu_rate_real = {}
arsr = 0
arsr_num = 0
for road in non_seed_road:
    if seed_num[road] != 0:
        accu_rate_real[road] = abs(v_est_all[road]-v_equal1[road])
        arsr += accu_rate_real[road]/v_equal1[road]
        arsr_num += 1
mape_real = arsr/arsr_num


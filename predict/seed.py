#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 20:12:25 2017

@author: qiushye
"""
import os
import copy
'''---data格式:road_id,start_id,end_id,length,level,time,v,date,weekday(),period,0/1---'''
def max_dic(dic):
    v = list(dic.values())
    k = list(dic.keys())
    return k[v.index(max(v))]

def min_dic(dic):
    v = list(dic.values())
    k = list(dic.keys())
    return k[v.index(min(v))]

def sort_size(dir,file_list):
    file_data = []
    for file in file_list:
        size = os.path.getsize(dir+file)
        file_data.append((file,size))
    file_data.sort(key = lambda l : l[1],reverse = True)
    res = [val[0] for val in file_data]
    return res
    

#选择时段
def choose_period(road_id_dir):
    period_count = {}
    all_period_count = {}
    #max_perod_count = {}
    #公共时段
    common_period = []
    for i in range(24):
        common_period.append(str(i))
        all_period_count[str(i)] = 0
    for file_name in os.listdir(road_id_dir):
        #data = []
        period = []
        for i in range(24):
            period_count[str(i)] = 0
            
        with open(road_id_dir+file_name,'r') as f:
            for line in f:
                set1 = line.strip().split(',')
                period_count[set1[9]] += 1
                all_period_count[set1[9]] += 1
        for i in range(24):
            if period_count[str(i)] > 16:
                period.append(str(i))
                
        common_period = [value for value in common_period if value in period]
        
        #print file_name
        #print period_count
    period_count_temp = {}
    for p in common_period:
        period_count_temp[p] = all_period_count[p]
    
    #print common_period
    return common_period[len(common_period)/2]
        #max_perod_count[max_period] += 1
    '''while 1:
        try:#所有路段信息具有公共时段
            if len(common_period) != 0:
                return common_period[0]
        except ValueError:
           print "There is no common period!"
           '''
    #print common_period
    #return min_dic(max_perod_count)

#选择10个工作日和4个周末
def choose_date(period,road_id_dir):
    
    #weekday_v_sum = 0
    common_weekday_date = []
    common_weekend_date = []
    #weekend_v_sum = 0
    weekday = ['1','2','3','4','5']
    weekend = ['6','0']
    #road_v_equal = {}
    #按文件大小排序
    file_list = os.listdir(road_id_dir)
    file_res = sort_size(road_id_dir,file_list)
    for file_name in file_res:
        weekday_date_count = []
        weekend_date_count = []
        with open(road_id_dir+file_name,'r') as f:
            for line in f:
                
                record = line.strip().split(',')
                if record[8] in weekday and record[7] not in weekday_date_count:
                    if int(record[9]) - int(period) == -1:
                        temp_v = float(record[6])
                    if record[9] == period:
                        weekday_date_count.append(record[7])
                        temp_v = 0
                    if int(record[9]) - int(period) == 1 and temp_v != 0 and float(record[6]) - temp_v < 4:
                        weekday_date_count.append(record[7])
                        
                    #weekday_v_sum += float(record[5])
                if record[8] in weekend and record[7] not in weekend_date_count:
                    if int(record[9]) - int(period) == -1:
                        temp_v = float(record[6])
                    if record[9] == period:
                        weekend_date_count.append(record[7])
                        temp_v = 0
                    if int(record[9]) - int(period) == 1 and temp_v != 0 and float(record[6]) - temp_v < 4:
                        weekend_date_count.append(record[7])
                    #weekend_v_sum += float(record[5])
                #weekday_v_equal = weekday_v_sum/
        
        if file_name == file_res[0]:
            common_weekday_date = weekday_date_count
            common_weekend_date = weekend_date_count
        
        
        
        common_weekday_date = [val for val in common_weekday_date if val in weekday_date_count]
        common_weekend_date = [val for val in common_weekend_date if val in weekend_date_count]
    return common_weekday_date[0:10],common_weekend_date[0:4]

#获得所有路段的信息
#roadseg_info:[{v_weekday},{delta_v_weekday},{v_weekend},{delta_v_weekend},v_equal_weekday,v_equal_weekend，dv_weekday,['1-hop'neighbor]]
#all_road_info:{road_id:roadseg_info,...}
def get_roadseg_info(period,road_id_dir,c_weekday_d,c_weekend_d,road_cross_dir):
    all_road_info = {}
    date = ''
    all_road = []
    #c_weekday_d,c_weekend_d = choose_date(period,road_id_dir)
    for file_name in os.listdir(road_id_dir):
        road_id = file_name.split('.')[0]
        all_road_info[road_id] = []
        all_road.append(road_id)
    for file_name in os.listdir(road_id_dir):
        #road_id = file_name.split('.')[0]
        weekday_v_sum = 0
        weekend_v_sum = 0
        v_equal_weekday = 0
        v_equal_weekend = 0
        roadseg_info = []
        delta_v_weekday = {}
        delta_v_weekend = {}
        v_weekday = {}
        v_weekend = {}
        dv_weekday = 0
        #v_temp_weekday = {}
        #v_temp_weekend = {}
        for date in c_weekday_d:
            v_weekday[date] = []
        for date in c_weekend_d:
            v_weekend[date] = []
        with open(road_id_dir+file_name,'r') as f:
            for line in f:
                record = line.strip().split(',')
                #速度保留一位小数
                v = round(float(record[6]),2)
                
                #print v
                if int(record[9]) - int(period) == -1:
                    temp_v = v
                if record[9] == period:# and record[6] in c_weekday_d:
                    #print file_name
                    #print record
                    #print c_weekday_d
                    date = record[7]
                    temp_v = 0
                    if date in c_weekday_d :
                        #print file_name
                        #print record
                        v_weekday[date] = v
                        weekday_v_sum += v
                    if date in c_weekend_d :
                        v_weekend[date] = v
                        weekend_v_sum += v
                if int(record[9]) - int(period) == 1 and temp_v != 0:
                    v = (temp_v + v)/2
                    date = record[7]
                    
                    if date in c_weekday_d :
                        #print file_name
                        #print record
                        v_weekday[date] = v
                        weekday_v_sum += v
                    if date in c_weekend_d :
                        v_weekend[date] = v
                        weekend_v_sum += v
        #if v_temp_weekday.values()[0] == []:
            #continue
        #print v_temp_weekday
        #print v_temp_weekend
        
        v_equal_weekday = weekday_v_sum/len(c_weekday_d)
        v_equal_weekend = weekend_v_sum/len(c_weekend_d)
        
        for date in c_weekday_d:
            if v_weekday[date] > v_equal_weekday:
                delta_v_weekday[date] = '+1'
            else:
                delta_v_weekday[date] = '-1'
            dv_weekday += (v_weekday[date]-v_equal_weekday)**2
        dv_weekday = dv_weekday/len(c_weekday_d)
        for date in c_weekend_d:
            if v_weekend[date] > v_equal_weekend:
                delta_v_weekend[date] = '+1'
            else:
                delta_v_weekend[date] = '-1'
        
        roadseg_info.append(v_weekday)
        roadseg_info.append(delta_v_weekday)
        roadseg_info.append(v_weekend)
        roadseg_info.append(delta_v_weekend)
        roadseg_info.append(round(v_equal_weekday,2))
        roadseg_info.append(round(v_equal_weekend,2))
        roadseg_info.append(round(dv_weekday,2))
        #print roadseg_info
        road_id = file_name.split('.txt')[0]
        all_road_info[road_id] = roadseg_info
    #找路段的1-hop,2-hop邻居
    
    #print all_road
    road_cross = {}
    with open(road_cross_dir,'r') as f:
        for line in f:
            set1 = line.strip().split(',')
            if set1[0] in all_road:
                road_cross[set1[0]] = set1[1:3]
    all_1hop_neighbor = {}
    all_2hop_neighbor = {}
    for road1 in all_road:
        all_1hop_neighbor[road1] = []
        all_2hop_neighbor[road1] = set()
    for road1 in all_road:
        neighbor_1 = []
        for road2 in list(road_cross.keys()):
            if road1 != road2:
                if road_cross[road1][0] in road_cross[road2] or road_cross[road1][1] in road_cross[road2]:
                    neighbor_1.append(road2)
        all_1hop_neighbor[road1] = neighbor_1
    '''
    for road1 in all_road:
        neighbor_2 = set()
        for road2 in all_1hop_neighbor[road1]:
            neighbor_2 = neighbor_2|set(all_1hop_neighbor[road2])
        all_2hop_neighbor[road1] = neighbor_2-set([road1])
'''
    for road_id in all_road:
        neighbor = set(all_1hop_neighbor[road_id])|set(all_2hop_neighbor[road_id])
        all_road_info[road_id].append(list(neighbor))
    return  all_road_info

#获得两个路段的相关系数,只考虑工作日
def get_road_R(road1,road2,all_road_info,c_weekday_d,c_weekend_d,threshold):
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
    if float(weekday_equal_rate)/float(len(c_weekday_d)) > threshold:# and float(weekend_equal_rate)/float(len(c_weekend_d)) > threshold:
        road1_R = 1
        road2_R = 1
    else:
        road1_R = 0
        road2_R = 0
    return road1_R,road2_R


#选择第一个种子
def choose_first_seed(all_road_info,period,c_weekday_d,c_weekend_d,threshold):
    #period = choose_period(road_id_dir)
    #all_road_info = get_roadseg_info(period,road_id_dir)
    all_road = list(all_road_info.keys())
    R = {}
    for road in all_road:
        R[road] = 0
    for road1 in all_road:
        for road2 in all_road:
            if road1 != road2:
                R1,R2 = get_road_R(road1,road2,all_road_info,c_weekday_d,c_weekend_d,threshold)
                R[road1] += R1
                R[road2] += R2
    #print R
    return max_dic(R)

#获得混合算法的增量
def Hybrid_rise(all_road_info,road,seed,c_weekday_d,c_weekend_d,threshold,alpha):
    
    all_road = list(all_road_info.keys())
    COV_last = 0
    COV_next = 0
    SUP_last = 0
    SUP_next = 0
    COV_road_last = []
    COV_road_next = []
    COV_last_group = []
    COV_next_group = []
    
    for road1 in all_road:
        if road1 not in seed and road1 not in COV_road_last:
            road1_neighbor = all_road_info[road1][-1]
            for seed1 in seed:
                if seed1 in road1_neighbor and get_road_R(road1,seed1,all_road_info,c_weekday_d,c_weekend_d,threshold) == (1,1):
                    COV_road_last.append(road1)
    COV_last_group = list(set(COV_road_last))
    COV_last = len(COV_last_group)             
                    
    for road2 in COV_last_group:
        road2_neighbor = all_road_info[road2][-1]
        for seed2 in seed:
            if seed2 in road2_neighbor and get_road_R(road2,seed2,all_road_info,c_weekday_d,c_weekend_d,threshold) == (1,1):
                SUP_last += 1
    seed_temp = copy.deepcopy(seed)
    seed_temp.append(road)
    #print seed_temp
    for road3 in all_road:
        if road3 not in seed_temp and road3 not in COV_road_next:
            road3_neighbor = all_road_info[road3][-1]
            for seed3 in seed_temp:
                if seed3 in road3_neighbor and get_road_R(road3,seed3,all_road_info,c_weekday_d,c_weekend_d,threshold) == (1,1):
                    COV_road_next.append(road3)
                    
    COV_next_group = list(set(COV_road_next))
    COV_next = len(COV_next_group)
    for road4 in COV_next_group:
        road4_neighbor = all_road_info[road4][-1]
        for seed4 in seed_temp:
            if seed4 in road4_neighbor and get_road_R(road4,seed4,all_road_info,c_weekday_d,c_weekend_d,threshold) == (1,1):
                SUP_next += 1   
    '''          
    for seed1 in seed:
        seed1_neighbor = all_road_info[seed1][-1]
        for road1 in seed1_neighbor:
            if road1 not in seed and road1 not in COV_road_last and get_road_R(road1,seed1,all_road_info,c_weekday_d,c_weekend_d,threshold) == (1,1):
                COV_road_last.append(road1)
                COV_last += 1
    for road2 in COV_road_last:
        road2_neighbor = all_road_info[road2][-1]
        for seed2 in seed:
            #road2_neighbor = all_road_info[road2][-1]
            if seed2 in road2_neighbor and get_road_R(road2,seed2,all_road_info,c_weekday_d,c_weekend_d,threshold) == (1,1):
                SUP_last += 1
                
    seed_temp = copy.deepcopy(seed)
    seed_temp.append(road)
    for seed3 in seed_temp:
        seed3_neighbor = all_road_info[seed3][-1]
        for road3 in seed3_neighbor:
            if road3 not in seed_temp and road3 not in COV_road_next and get_road_R(road3,seed3,all_road_info,c_weekday_d,c_weekend_d,threshold) == (1,1):
                COV_road_next.append(road3)
                COV_next += 1
    for road4 in COV_road_next:
        road4_neighbor = all_road_info[road4][-1]
        for seed4 in seed_temp:
            #road2_neighbor = all_road_info[road2][-1]
            if road4 in road4_neighbor and get_road_R(road4,seed4,all_road_info,c_weekday_d,c_weekend_d,threshold) == (1,1):
                SUP_next += 1
    '''
    return (COV_next+SUP_next)-alpha*(COV_last+SUP_last)
                
                
#种子选择
def seed_select(road_id_dir,threshold,seed_rate,alpha):
    #data:某个路段各时刻的速度数据：v,period,date,weekday(road_id.txt)
    global all_road_info,period,c_weekday_d,c_weekend_d
    road_cross_dir = "/home/qiushye/result_choose.txt"
    #threshold = 0.7
    #seed_rate = 0.15
    #alpha = 1
    SEED = []
    period = choose_period(road_id_dir)
    
    c_weekday_d,c_weekend_d = choose_date(period,road_id_dir)
    #print c_weekday_d,c_weekend_d
    
    all_road_info = get_roadseg_info(period,road_id_dir,c_weekday_d,c_weekend_d,road_cross_dir)
    first_seed = choose_first_seed(all_road_info,period,c_weekday_d,c_weekend_d,threshold)
    SEED.append(first_seed)
    all_road = list(all_road_info.keys())
    #print len(all_road)
    hybrid_rise_set = {}
    #for road in all_road:
        #hybrid_rise_set[road] = 0
    SEED_num = int(seed_rate*len(all_road))
    for road1 in all_road:
            #if road1 not in SEED:
                hybrid_rise_set[road1] = 0
    for i in range(SEED_num):
        #new_seed = ''
        #hybrid_rise_set = {}
        
        
        for road in all_road:
            #保证种子的方差小于10
            if road not in SEED and all_road_info[road][-2] < 10:
                hybrid_rise_set[road] = Hybrid_rise(all_road_info,road,SEED,c_weekday_d,c_weekend_d,threshold,alpha)
                
        #print SEED
        
        new_seed = max_dic(hybrid_rise_set)
        if hybrid_rise_set[new_seed] == 0:
            break
        
        SEED.append(new_seed)
        hybrid_rise_set[new_seed] = 0
        #print hybrid_rise_set
        #if float(len(SEED))/float(len(all_road)) >= 0.15:
            #break
    return SEED
'''
threshold = 0.7
seed_rate = 0.15
alpha = 1
road_id_dir = "/home/qiushye/road_id_division_test/"
SEED =  seed_select(road_id_dir,threshold,seed_rate,alpha)
'''       
            
    




            
    
                   
        
        
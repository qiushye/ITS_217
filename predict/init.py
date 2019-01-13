'''
数据、结果目录的初始化
日期、工作日周末及索引的设定
'''
import os
import datetime

cur_dir = os.path.split(os.path.realpath(__file__))[0]
data_dir = cur_dir + '/data/'
result_dir = cur_dir + '/result/'

dates = [
    '2012-11-07', '2012-11-08', '2012-11-09', '2012-11-10', '2012-11-11',
    '2012-11-12', '2012-11-13', '2012-11-14', '2012-11-15', '2012-11-16',
    '2012-11-17', '2012-11-18', '2012-11-19', '2012-11-20', '2012-11-21'
]

weekday, weekend = [], []
weekday_index, weekend_index = [], []
for i in range(len(dates)):
    date = dates[i]
    [y, m, d] = list(map(int, date.split('-')))
    dt = datetime.date(y, m, d)
    if dt.weekday() < 5:
        weekday.append(date)
        weekday_index.append(i)
    else:
        weekend.append(date)
        weekend_index.append(i)

# print(weekday, weekend)

# dates = dates[:8]
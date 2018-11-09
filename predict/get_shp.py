#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:13:32 2017

@author: qiushye
"""

import shapefile

f = open("/home/qiushye/road_choose.txt", 'r')
all_road = []

for line in f:
    rec = line.strip()
    all_road.append(str(rec))

SEED2 = ['59567211760',
         '59567217347',
         '59567202858',
         '59567216285',
         '59567216282',
         '59567214702',
         '59567213355',
         '59567218015',
         '59567213236',
         '59567214775',
         '59567215346',
         '59567214780',
         '59567214794',
         '59567214770']

SEED1 = ['59567214700',
         '59567202858',
         '59567217347',
         '59567214702',
         '59567211760',
         '59567213355',
         '59567218015',
         '59567202818',
         '59567213236',
         '59567202945',
         '59567215346',
         '59567214445',
         '59567216282',
         '59567214704']

SEED3 = ['59567211554',
         '59567209884',
         '59567216282',
         '59567202858',
         '59567216227',
         '59567205374',
         '59567217348',
         '59567216285',
         '59567214702',
         '59567214794',
         '59567213355',
         '59567218015',
         '59567214793',
         '59567215346']
sf = shapefile.Reader("/home/qiushye/experiment_map/road_choose.shp")

Records = sf.records()
shapesRecords = sf.shapeRecords()
fields = sf.fields

w = shapefile.Writer(shapeType=3)
w.autoBlance = 1
for f in fields:
    if f != fields[0]:
        w.field(f[0], f[1], f[2], f[3])
    # if f == fields[3]:
        # break

for road in SEED3:
    for rec in Records:
        if road == rec[2]:
            index = Records.index(rec)
            points = shapesRecords[index].shape.points
            # for point in points:
            # w.point(point[0],point[1])
            w.line(parts=[[[val[0], val[1]] for val in points]])
            w.record(rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], rec[7], rec[8], rec[9], rec[10], rec[11],
                     rec[12], rec[13], rec[14], rec[15], rec[16], rec[17], rec[18], rec[19], rec[20], rec[21])

w.save("/home/qiushye/seed.shp")

sf = shapefile.Reader("/home/qiushye/experiment_map/road_choose_2.shp")

Records = sf.records()
with open("/home/qiushye/write.txt", "w+") as fw:
    for rec in Records:
        fw.write(rec[2]+'\n')

import numpy as np
import pandas as pd
import csv
import sys
import datetime
from math import radians, cos, sin, asin, sqrt

def print_msg(*args,end='\n'):
    for item in args:
        sys.stdout.write(str(item)+' ')
    sys.stdout.write(end)
    sys.stdout.flush()

def qick_distance(Lat1, Long1, Lat2, Long2):
    x = Lat2 - Lat1
    y = (Long2 - Long1)*cos((Lat2 + Lat1)*0.00872664626)  
    return 111.138*sqrt(x*x+y*y)

file = open('stations_coordinates.csv')
csvreader = csv.reader(file)

header = next(csvreader)

data = []
for row in csvreader:
    data.append(row)

first = data[0]
second = data[1]

result_file = open('stations_dist.csv', 'w', newline='')
writer = csv.writer(result_file)

for i , station in enumerate(data):
    for j, other_station in enumerate(data):
        if i != j:
            writer.writerow([station[0],other_station[0], qick_distance(float(station[1]), float(station[2]), float(other_station[1]), float(other_station[2]))])
    
result_file.close()
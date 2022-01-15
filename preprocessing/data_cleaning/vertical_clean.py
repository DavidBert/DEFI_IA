import numpy as np
import pandas as pd
import csv
import sys
import datetime

def print_msg(*args,end='\n'):
    for item in args:
        sys.stdout.write(str(item)+' ')
    sys.stdout.write(end)
    sys.stdout.flush()

data = pd.read_csv('X_station_train.csv')

file = open('X_station_train.csv')
csvreader = csv.reader(file)

header = next(csvreader)

data = []
for row in csvreader:
    #print_msg(row)
    data.append(row)

newChenk = True
index = 0
chenk = []
newData = []

result_file = open('results.csv', 'w', newline='')
writer = csv.writer(result_file)
newData = []
count = 1000
for row in data:
    if newChenk :
        count = count -1 
        if count == 0 :
            count = 1000
            print_msg(row)
        chenk = []
        station = row[0]
        date = row[1]
        index = 1
        newChenk = False
        chenk.append(row)
    else:
        index = index + 1
        if station == row[0] :
            chenk.append(row)     
        else:
            newChenk =True
            if index == 24 :
                newData = newData + chenk
                #print_msg('************************************************************************************')
file.close()
print_msg('save data')
for row in newData:
    writer.writerow(row)
result_file.close()


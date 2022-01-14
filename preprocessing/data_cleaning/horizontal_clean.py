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

def find_row(station, date, data, neighbors):
    nei = None
    for row in neighbors:
        if row[0] == station:
            nei = row
    if nei == None:
        return None 

    for row in data:
        if row[0] == nei[1] and row[1] == date:
            return row
        if row[0] == nei[2] and row[1] == date:
            return row
        if row[0] == nei[3] and row[1] == date:
            return row
        if row[0] == nei[4] and row[1] == date:
            return row
        if row[0] == nei[5] and row[1] == date:
            return row
    return None

file = open('results.csv')
csvreader = csv.reader(file)

header = next(csvreader)

data = []
for row in csvreader:
    #print_msg(row)
    data.append(row)
file.close()


file = open('neighbor.csv')
csvreader = csv.reader(file)

header = next(csvreader)

neighbors = []
for row in csvreader:
    neighbors.append(row)
file.close()

result_file = open('v_cleaned_data.csv', 'w', newline='')
writer = csv.writer(result_file)

count = 1000
for row in data:
    nb_row = find_row(row[0], row[1], data, neighbors)
    if nb_row == None:
        print_msg('ERROR station',row[0])
    else:  
        if row[2] == '':
            row[2] = nb_row[2]
        if row[3] == '':
            row[3] = nb_row[3]
        if row[4] == '':
            row[4] = nb_row[4]
        if row[5] == '':
            row[5] = nb_row[5]
        if row[6] == '':
            row[6] = nb_row[6]
        if row[7] == '':
            row[7] = nb_row[7]

        count = count - 1
        if count == 0 :
            count = 1000
            print_msg(row)
    
    writer.writerow(row)

result_file.close()
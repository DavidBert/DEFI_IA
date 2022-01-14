import numpy as np
import pandas as pd
import csv
import sys

def print_msg(*args,end='\n'):
    for item in args:
        sys.stdout.write(str(item)+' ')
    sys.stdout.write(end)
    sys.stdout.flush()

def prec(current):
    prec = current - 1
    if prec == 0 :
        prec = 23
    return prec


file = open('X_station_test_cleaned.csv')
csvreader = csv.reader(file)

header = next(csvreader)

data = []
for row in csvreader:
    #print_msg(row)
    data.append(row)
first = True
for row in data:
    #print_msg(row)
    if first :
        pre = int(row[11])
        first = False
    else:
        current  = int(row[11])
        if prec(current) != pre:
            print_msg(row)
            break
        else:
            pre = current


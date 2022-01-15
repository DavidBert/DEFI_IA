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

data = pd.read_csv('results.csv')
data = data.drop('Id', axis=1)
data['month'] = pd.DatetimeIndex(data['date']).month


for i in range(0, data.shape[0], 23):
    tmp = data[i:i+23]
    tmp = tmp.reset_index()
    date_time_start = datetime.datetime.strptime(tmp.iat[0,2], '%Y-%m-%d %H:%M:%S')
    date_time_end = datetime.datetime.strptime(tmp.iat[22,2], '%Y-%m-%d %H:%M:%S')
    if date_time_start.hour != 1 or date_time_end.hour != 23  :
        print_msg(tmp.iat[0,1],tmp.iat[0,2],tmp.iat[22,2])
        break





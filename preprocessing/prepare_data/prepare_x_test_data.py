import numpy as np
import pandas as pd
import csv
import sys


def print_msg(*args,end='\n'):
    for item in args:
        sys.stdout.write(str(item)+' ')
    sys.stdout.write(end)
    sys.stdout.flush()


# Read X_station_test and Id_month_test

station_test = pd.read_csv('X_station_test.csv')
Id_month_test = pd.read_csv('Id_month_test.csv')

station_test['station'] =  station_test['Id'].str.split('_').str[0]
station_test['day_index'] =  station_test['Id'].str.split('_').str[1]
station_test['hour'] =  station_test['Id'].str.split('_').str[2]


# Replace NaNs by mean
station_test['ff'] = station_test['ff'].fillna((station_test['ff'].mean()))
station_test['t'] = station_test['t'].fillna((station_test['t'].mean()))
station_test['td'] = station_test['td'].fillna((station_test['td'].mean()))
station_test['hu'] = station_test['hu'].fillna((station_test['hu'].mean()))
station_test['dd'] = station_test['dd'].fillna((station_test['dd'].mean()))
station_test['precip'] = station_test['precip'].fillna((station_test['precip'].mean()))


station_test = station_test[station_test.hour != '0' ]

station_test.to_csv(r'X_station_test_cleaned.csv')

print(station_test.head())
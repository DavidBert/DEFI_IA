
"""

This program computes the error for each station, each date using the forecasts data and the groud truth data

"""

import numpy as np
import pandas as pd
import csv


def print_msg(*args,end='\n'):
    for item in args:
        sys.stdout.write(str(item)+' ')
    sys.stdout.write(end)
    sys.stdout.flush()

# Read Y_forecast and Y_ground_truth

Y_forecast = pd.read_csv('Y_forecast.csv')
Y_ground_truth = pd.read_csv('Y_ground_truth.csv')


# compute errors
dates = []
stations = []
errors  = []
for index, gt in Y_ground_truth.iterrows():
    fc = Y_forecast.loc[(Y_forecast['station'] == gt['number_sta']) & (Y_forecast['date'] == gt['date'])]
    dates.append(gt['date'])
    stations.append(gt['number_sta'])
    errors.append(float(fc['forecast']) - float(gt['Ground_truth']))
    print(index, float(fc['forecast']) - float(gt['Ground_truth']))

# save errors as csv file

y_error = pd.DataFrame({'date': dates, 'station': stations, 'error': errors})
y_error.to_csv(r'y_error.csv')

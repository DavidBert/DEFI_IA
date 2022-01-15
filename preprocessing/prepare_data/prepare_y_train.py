import numpy as np
import pandas as pd
import csv
from numpy import savetxt
from datetime import datetime
from datetime import timedelta

data = pd.read_csv('X_data.csv')

Y_data = pd.read_csv('y_error.csv')
Y_data['error'] = Y_data['error'].fillna(0)

np_data = np.array(data)
y_train = []
for i in range(0, np_data.shape[0] - 1, 23):
    tmp = np_data[i:i+23]
    if tmp[0,0] == '' or tmp[0,1] == '':
        print("ERROR")
    date_time_obj = datetime.strptime(tmp[0,1], '%Y-%m-%d %H:%M:%S')
    date_time_obj = date_time_obj + timedelta(days=1) 
    rslt_df = Y_data.loc[(Y_data['station'] == tmp[0,0]) & (Y_data['date'] == date_time_obj.strftime("%Y-%m-%d"))]
    a = np.array(rslt_df['error'])
    print(date_time_obj.strftime("%Y-%m-%d"),tmp[0,0],a[0])
    y_train.append(a[0])
y_train

savetxt('y_train.csv', y_train, delimiter=',')


# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:24:27 2022

@author: Noemi
"""

import csv
import pandas as pd
import numpy as np



def train_feature(file):
    
    # Data_paths
    X_station = pd.read_csv(file,sep=",",header=0)
    # add cos and sin of month
    
    X_station['cosmonth'] = np.cos((X_station['month']-1)*2*np.pi/12)
    X_station['sinmonth'] = np.sin((X_station['month']-1)*2*np.pi/12)
    
    #Delete unseless rows
    del X_station['month']
    del X_station['Unnamed: 0']
    del X_station['id']

    # Copy of X_station to put the columns in alphabetical order
    X_station_cpy = X_station.sort_index(axis=1)
    del X_station_cpy['sinmonth']
    del X_station_cpy['cosmonth']
    
    # Compute the mean for each variables 
    d2m = pd.DataFrame(X_station_cpy[X_station_cpy.columns[:24]].mean(axis=1),columns=['d2m_mean'])
    dd = pd.DataFrame(X_station_cpy[X_station_cpy.columns[24:48]].mean(axis=1),columns=['dd_mean'])
    ff = pd.DataFrame(X_station_cpy[X_station_cpy.columns[48:72]].mean(axis=1),columns=['ff_mean'])
    hu = pd.DataFrame(X_station_cpy[X_station_cpy.columns[72:96]].mean(axis=1),columns=['hu_mean'])
    msl = pd.DataFrame(X_station_cpy[X_station_cpy.columns[96:120]].mean(axis=1),columns=['msl_mean'])
    p3031 = pd.DataFrame(X_station_cpy[X_station_cpy.columns[120:144]].mean(axis=1),columns=['p3031_mean'])
    precip = pd.DataFrame(X_station_cpy[X_station_cpy.columns[144:168]].sum(axis=1),columns=['precip_sum'])
    r = pd.DataFrame(X_station_cpy[X_station_cpy.columns[168:192]].mean(axis=1),columns=['r_mean'])
    t2m = pd.DataFrame(X_station_cpy[X_station_cpy.columns[192:216]].mean(axis=1),columns=['t2m_mean'])
    t = pd.DataFrame(X_station_cpy[X_station_cpy.columns[216:240]].mean(axis=1),columns=['t_mean'])
    td = pd.DataFrame(X_station_cpy[X_station_cpy.columns[240:264]].mean(axis=1),columns=['td_mean'])
    tp = pd.DataFrame(X_station_cpy[X_station_cpy.columns[264:288]].sum(axis=1),columns=['tp_sum'])
    u10 = pd.DataFrame(X_station_cpy[X_station_cpy.columns[288:312]].mean(axis=1),columns=['u10_mean'])
    v10 = pd.DataFrame(X_station_cpy[X_station_cpy.columns[312:336]].mean(axis=1),columns=['v10_mean'])
    ws = pd.DataFrame(X_station_cpy[X_station_cpy.columns[336:360]].mean(axis=1),columns=['ws_mean'])

    #Merge all dataframes
    X_station_mean = pd.concat([d2m, dd,ff, hu, msl,p3031,precip,r,t2m,t,td,tp,u10,v10,ws, X_station[['sinmonth','cosmonth']]], axis=1)


    return X_station_mean

def test_feature(file1):
    
    X_station_test = pd.read_csv(file1,sep=",",header=0)

    del X_station_test['Unnamed: 0']
    X_station_test["cosmonth"]=np.cos((2*np.pi*(X_station_test["month"]-1))/12)
    X_station_test["sinmonth"]=np.sin((2*np.pi*(X_station_test["month"]-1))/12)
    del X_station_test["month"]



    X_station_test_cpy = X_station_test.sort_index(axis=1)
    del X_station_test_cpy['sinmonth']
    del X_station_test_cpy['cosmonth']
    del X_station_test_cpy['id']
    
    # Compute the mean for each variables 
    d2m = pd.DataFrame(X_station_test_cpy[X_station_test_cpy.columns[:24]].mean(axis=1),columns=['d2m_mean'])
    dd = pd.DataFrame(X_station_test_cpy[X_station_test_cpy.columns[24:48]].mean(axis=1),columns=['dd_mean'])
    ff = pd.DataFrame(X_station_test_cpy[X_station_test_cpy.columns[48:72]].mean(axis=1),columns=['ff_mean'])
    hu = pd.DataFrame(X_station_test_cpy[X_station_test_cpy.columns[72:96]].mean(axis=1),columns=['hu_mean'])
    msl = pd.DataFrame(X_station_test_cpy[X_station_test_cpy.columns[96:120]].mean(axis=1),columns=['msl_mean'])
    p3031 = pd.DataFrame(X_station_test_cpy[X_station_test_cpy.columns[120:144]].mean(axis=1),columns=['p3031_mean'])
    precip = pd.DataFrame(X_station_test_cpy[X_station_test_cpy.columns[144:168]].sum(axis=1),columns=['precip_sum'])
    r = pd.DataFrame(X_station_test_cpy[X_station_test_cpy.columns[168:192]].mean(axis=1),columns=['r_mean'])
    t2m = pd.DataFrame(X_station_test_cpy[X_station_test_cpy.columns[192:216]].mean(axis=1),columns=['t2m_mean'])
    t = pd.DataFrame(X_station_test_cpy[X_station_test_cpy.columns[216:240]].mean(axis=1),columns=['t_mean'])
    td = pd.DataFrame(X_station_test_cpy[X_station_test_cpy.columns[240:264]].mean(axis=1),columns=['td_mean'])
    tp = pd.DataFrame(X_station_test_cpy[X_station_test_cpy.columns[264:288]].sum(axis=1),columns=['tp_sum'])
    u10 = pd.DataFrame(X_station_test_cpy[X_station_test_cpy.columns[288:312]].mean(axis=1),columns=['u10_mean'])
    v10 = pd.DataFrame(X_station_test_cpy[X_station_test_cpy.columns[312:336]].mean(axis=1),columns=['v10_mean'])
    ws = pd.DataFrame(X_station_test_cpy[X_station_test_cpy.columns[336:360]].mean(axis=1),columns=['ws_mean'])
    
    #Merge all dataframes
    X_station_test_mean = pd.concat([d2m, dd,ff,hu,msl,p3031,precip,r,t2m,t,td,tp,u10,v10,ws, X_station_test[['sinmonth','cosmonth']]], axis=1)
    
    
    return X_station_test_mean
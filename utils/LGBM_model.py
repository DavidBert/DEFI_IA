# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:10:02 2022

@author: Noemi
"""
import csv
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


import tensorflow as tf
import lightgbm as lgb
import tensorflow.keras.optimizers as ko
import tensorflow.keras.backend as k
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl

import tensorflow.keras.applications as ka
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, concatenate
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import  make_scorer, mean_absolute_percentage_error, r2_score 
from sklearn.model_selection import cross_validate
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import os
import argparse
#import feature_engineering as fe


 
def modele(file1,file2,file3):

    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', help='Folder to data files')
    parser.add_argument(
        '--output_folder', help='Folder to save the model, training logs, and predictions',
    )
    args = parser.parse_args()

    data_path = args.data_path
    results_path = args.output_folder

    for path in [data_path, results_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    #Split train/test
    X_train, X_test, y_train, y_test = train_test_split(file1,file2, test_size=0.25, random_state=34)
    
    #Scale the dataset
    scaler = StandardScaler(with_mean=True,with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #take only the variable of interest
    y_train = np.array(y_train['Ground_truth'])
    y_test = np.array(y_test['Ground_truth'])
    
    #Define the MAPE loss
    def def_MAPE(y_true, y_pred): 
        ''' Args : y_pred = forecast 
                   y_true = actual 
                   
                   !!!!!! y_pred and y_pred must be numpy arrays !!!!!!
                   
                   Return : the opposite of the loss function (MAPE)
           ''' 
        y_true = y_true
        y_pred = y_pred
        n = len(y_true)
        return  - 100/n * np.sum(np.abs((y_true-y_pred)/y_true)) # MAPE is a loss function, so we multiply by -1
    # Structure of the MLP model
    print("running for 30.000 iterations")
    lgb_train=lgb.Dataset(X_train_scaled, label=y_train+1)
    lgb_eval = lgb.Dataset(X_test_scaled, label=y_test+1, reference=lgb_train)
    params = {'boosting_type': 'gbdt',
    'objective': 'mape',
    'feature_fraction': 0.8,
    'max_depth': 8,
    'num_leaves' : 32,
    'learning_rate' : 0.001,
    'n_estimators' : 100,
    'min_child_samples' : 100,
    'early_stopping' : 2000,
    'reg_alpha' : 0,
    'reg_lambda' : 0,
    'bagging_fraction' :0.7,
    'metric': 'mape'}

    gbm_regressor = lgb.train(params,
    lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=['train','valid'],)
   
    #y_pred=gbm_regressor.predict(X_test)
    #mape_local= def_MAPE(y_test+1, y_pred)
    
    
    
    
    path_parent = (os.getcwd())
    path_data='Data/Test/Test/Baselines/'
    path_baseline_test = os.path.join(path_parent,path_data).replace("/","\\")
    Baseline_obs_test = pd.read_csv(path_baseline_test+"Baseline_observation_test.csv",sep=",",header=0)
    X_station_test = pd.read_csv(path_baseline_test+"full_X_test.csv",sep=",",header=0)
   


    #Scale the testing set
    X_station_test_scaled = scaler.transform(file3)

    #get ids that we want to predict 
    ids = pd.DataFrame(Baseline_obs_test['Id'],columns=['Id'])
    #X_station_test_scaled.shape
   
    a=gbm_regressor.predict(X_station_test_scaled)
    pred_kaggle = pd.DataFrame() 
    pred_kaggle["Id"] = X_station_test['id'] 
    pred_kaggle["Prediction"] = a
    
    predictions = ids.merge(pred_kaggle,on="Id")
    predictions['Prediction'] = np.round(predictions['Prediction'],1) 
    predictions.loc[predictions['Prediction']<1.0, 'Prediction']=1.0
    if not(os.path.exists(results_path)):
        os.mkdir(results_path)
    predictions.to_csv(os.path.join(results_path,'predictions_lgbm_means.csv'),sep=',',index=False)#changer le path
    

import argparse
from ast import arg
import pandas as pd 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

## Import the custom functions 
from functions import Read_Parquet, spliting, season,wind_vx_vy,day_id,percentile,training_kfold,objective,run

from scipy.stats import kurtosis
from scipy.stats import skew

## Modelization
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb


## Hyperparametres tuning
import optuna 
from optuna import Trial, visualization
from optuna.samplers import TPESampler

## Model saving ( serialization)
import joblib


## agparse arguments

import argparse
train_path = '../input/imputed_data/final_imputed_train_x_station.gzip'
test_path = '../input/imputed_data/final_imputed_test_x_station.gzip'
month_id_path = '../input/Test/X_stations/Id_month_test.csv'
output_data_path = '../input/imputed_data/'
y_train_path = '../input/Train/X_stations/Y_train.csv'
coords_path = '../input/other/stations_coordinates.csv'
base_for_test_path = "../input/Test/Baselines/Baseline_forecast_test.csv"
base_for_train_path = "../input/Train/Baselines/Baseline_forecast_train.csv"




if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default=train_path, help='Path to imputed_data_train')
    parser.add_argument('--output_folder', type=str, default=output_data_path, help='Path to output_folder')

    args = parser.parse_args()

    data_path = args.data_path
    output_folder = args.output_folder

    print("Reading data")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")

    train = Read_Parquet(train_path)
    test = Read_Parquet(test_path)
    month_id = pd.read_csv(month_id_path)
    coords = pd.read_csv(coords_path)
    y_train = pd.read_csv(y_train_path)

    print("Preprocessing and engineering the data")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")


    train.height_sta = train.height_sta.astype('int16')
    test['Id'] = test['Id'].astype(str)
    test['day_index'] = test['Id'].apply(lambda x : spliting(x, 1))
    test['Hour'] = test['Id'].apply(lambda x : spliting(x, 2))
    test['number_sta'] = test['Id'].apply(lambda line : spliting(line,0)).astype('int64')
    test.Hour = test.Hour.astype('int16')
    test.height_sta = test.height_sta.astype('int16')

    month_id = month_id.astype(str)
    dictionnary = dict()
    for idx, val in month_id.values:
        dictionnary[idx] = val


    test['Month'] = test['day_index'].map(lambda x : dictionnary[x]).astype('int16')

    train["Month"] = train["date"].dt.month.astype('int16')
    train["season"] = train['Month'].apply(season)

    test['season'] = test['Month'].apply(season)

    test = wind_vx_vy(test)
    train = wind_vx_vy(train)



    train["Index"] = train["Id"].apply(day_id)
    test["Index"] = test["Id"].apply(day_id)
    train["date"] = train["date"].dt.date
    train[['lat','lon','height_sta','Month']] = train[['lat','lon','height_sta','Month']].astype('int16')
    test[['lat','lon','height_sta','Month']] = test[['lat','lon','height_sta','Month']].astype('int16')

    print("Aggregating of the training data")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")


    train_agg = train.groupby(['Index']).agg(
                                            {'temp': ['min', 'max','mean','median',percentile(25),percentile(75), skew, kurtosis],
                                             'humidity': ['min', 'max','mean','median',percentile(25),percentile(75), skew, kurtosis],
                                             'dew_point_temp': ['min', 'max','mean','median',percentile(25),percentile(75),skew, kurtosis],
                                             'Vx': ['min', 'max','mean','median',percentile(25),percentile(75),skew, kurtosis],
                                             'Vy': ['min', 'max','mean','median',percentile(25),percentile(75),skew, kurtosis],
                                             'precip': ['sum','min', 'max','mean','median',percentile(25),percentile(75),skew, kurtosis],
                                             'Month':pd.Series.unique,
                                             'season':pd.Series.unique,
                                             'number_sta':pd.Series.unique,# for further check 
                                             'date':pd.Series.unique,# for further check
                                             })



    train_agg.columns = ["_".join(x) for x in train_agg.columns.ravel()]

    train_agg.rename(columns={"Month_unique":"Month","season_unique":"season",	"number_sta_unique":"number_sta",	"date_unique":"date"},inplace=True)

    train_agg[["Month","season"]] = train_agg[["Month","season"]].astype('category')
    print("Aggregating of the test data")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")


    test_agg = test.groupby(['Index']).agg(
                                             {'temp': ['min', 'max','mean','median',percentile(25),percentile(75), skew, kurtosis],
                                             'humidity': ['min', 'max','mean','median',percentile(25),percentile(75), skew, kurtosis],
                                             'dew_point_temp': ['min', 'max','mean','median',percentile(25),percentile(75),skew, kurtosis],
                                             'Vx': ['min', 'max','mean','median',percentile(25),percentile(75),skew, kurtosis],
                                             'Vy': ['min', 'max','mean','median',percentile(25),percentile(75),skew, kurtosis],
                                             'precip': ['sum','min', 'max','mean','median',percentile(25),percentile(75),skew, kurtosis],
                                             'Month':pd.Series.unique,
                                             'season':pd.Series.unique,
                                             'number_sta':pd.Series.unique,
                                             })

    print("Processing the data after aggregation")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")

    test_agg.columns = ["_".join(x) for x in test_agg.columns.ravel()]



    test_agg.rename(columns={"Month_unique":"Month","season_unique":"season",	"number_sta_unique":"number_sta"},inplace=True)


    test_agg[["Month","season"]] = test_agg[["Month","season"]].astype('category')

    train_merge = train_agg.reset_index().merge(coords, on=['number_sta'], how='left')


    test = test_agg.reset_index().merge(coords, on=['number_sta'], how='left')


    y_train.rename(columns={'Id':'Index','date':'target_date','number_sta':'target_number_sta'}, inplace=True)
    y_train.set_index('Index', inplace=True)

    train = pd.concat([train_merge.set_index('Index'),y_train],verify_integrity=True, axis=1)

    test.set_index("Index", inplace=True)

    train_enc = pd.get_dummies(train,prefix_sep='_')

    test_enc = pd.get_dummies(test,prefix_sep='_')
    base_for_train = pd.read_csv(base_for_train_path)
    base_for_train.drop(['number_sta','date'], axis=1, inplace=True)
    base_for_train.rename(columns={'Prediction':'Prediction_meteo_france'}, inplace=True)

    base_for_test = pd.read_csv(base_for_test_path)
    base_for_test.rename(columns={'Prediction':'Prediction_meteo_france'}, inplace=True)

    test_enc.reset_index(inplace=True)
    test_enc.rename(columns={'Index':'Id'}, inplace=True)
    test= pd.merge(base_for_test, test_enc, how='right').set_index("Id")

    train_enc.reset_index(inplace=True)
    train_enc.rename(columns={'Index':'Id'}, inplace=True)
    train= pd.merge(base_for_train, train_enc, how='right').set_index("Id")
    
    train.fillna(train.median(), inplace=True)
    test.fillna(test.median(), inplace=True)

    print("Training xgboost model with cross validation ")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")


    KfoldsDf_train = training_kfold(train, kfold=10)

    xgboost = xgb.XGBRegressor(n_estimators=100,reg_lambda=1,max_depth=23, n_jobs=-1,reg_alpha=3, min_child_weight= 3, gamma=5, learning_rate=0.01, colsample_bytree=1.0)

    if __name__ == "__main__": 
        errors = []

        for i in range(10):
            score, model_name, fold = run(KfoldsDf_train, i,  xgboost ,mean_absolute_percentage_error)
            print(f"The model : {model_name} has a MAPE of {score} in fold {fold}")

            errors.append(score)
    

        print(f"The mean MAPE of all the folds is {np.mean(errors)}")

    print("Hyper parametres tuning OPTUNA")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")
    print("######################################################################################################################")

    y_train = KfoldsDf_train[KfoldsDf_train["kfold"]!=fold]['Ground_truth'].values
    X_train = KfoldsDf_train[KfoldsDf_train["kfold"]!=fold].drop(['kfold',"Ground_truth"], axis=1).set_index('Id')

    study = optuna.create_study(direction='minimize',sampler=TPESampler(), )
    study.optimize(lambda trial : objective(trial,X_train,y_train),n_trials= 10)

    print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))

    visualization.plot_optimization_history(study)

    visualization.plot_slice(study)

    visualization.plot_parallel_coordinate(study)

    visualization.plot_contour(study)

    



    
    





   

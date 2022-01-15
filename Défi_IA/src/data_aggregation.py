import pandas as pd
import numpy as np

from scipy.stats import kurtosis
from scipy.stats import skew

import os

test_path = '../input/imputed_data/first_eng_final_imputed_test_x_station.gzip'
train_path = '../input/imputed_data/first_eng_final_imputed_train_x_station.gzip'
coords_path = '../input/other/stations_coordinates.csv'
y_train_path = '../input/Train/X_stations/Y_train.csv'


def Read_Parquet(chemin):
  return pd.read_parquet(chemin, engine='fastparquet')

def day_id(line):
    line = str(line)
    sta = line.split('_')[0]
    day = line.split('_')[1]
    return "_".join([sta, day])

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

if __name__=='__main__':

    train = Read_Parquet(train_path)
    test= Read_Parquet(test_path)
    coords = pd.read_csv(coords_path)
    y_train = pd.read_csv(y_train_path)




    train["Index"] = train["Id"].apply(day_id)
    test["Index"] = test["Id"].apply(day_id)
    train["date"] = train["date"].dt.date
    train[['lat','lon','height_sta','Month']] = train[['lat','lon','height_sta','Month']].astype('int16')
    test[['lat','lon','height_sta','Month']] = test[['lat','lon','height_sta','Month']].astype('int16')


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

    test_agg.columns = ["_".join(x) for x in test_agg.columns.ravel()]



    test_agg.rename(columns={"Month_unique":"Month","season_unique":"season",	"number_sta_unique":"number_sta"},inplace=True)


    test_agg[["Month","season"]] = test_agg[["Month","season"]].astype('category')

    train_merge = train_agg.reset_index().merge(coords, on=['number_sta'], how='left')


    test_merge = test_agg.reset_index().merge(coords, on=['number_sta'], how='left')


    y_train.rename(columns={'Id':'Index','date':'target_date','number_sta':'target_number_sta'}, inplace=True)
    y_train.set_index('Index', inplace=True)

    train_concat = pd.concat([train_merge.set_index('Index'),y_train],verify_integrity=True, axis=1)

    train_concat.dropna(subset=['Ground_truth'], inplace=True)

    train_concat.drop(['number_sta','target_date','target_number_sta','date'], axis=1, inplace=True)


    test_merge.drop(['number_sta'], axis=1, inplace=True)

    train_concat.to_parquet(os.path.join('C:/Users/alaek/DEFI_IA/france_meteo/Défi_IA/input/imputed_data/',"X_Train_concat_y_train.gzip"), compression='gzip',engine='fastparquet')
    test_merge.to_parquet(os.path.join('C:/Users/alaek/DEFI_IA/france_meteo/Défi_IA/input/imputed_data/',"X_Test_merged_clean.gzip"), compression='gzip',engine='fastparquet')







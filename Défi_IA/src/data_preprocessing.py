import pandas as pd
import numpy as np
import sklearn
from tqdm import tqdm 
import math
import os


train_path = '../input/imputed_data/final_imputed_train_x_station.gzip'
test_path = '../input/imputed_data/final_imputed_test_x_station.gzip'
month_id_path = '../input/Test/X_stations/Id_month_test.csv'
output_data_path = '../input/imputed_data/'




def Read_Parquet(chemin):
  return pd.read_parquet(chemin, engine='fastparquet')

def spliting(string, index):
    liste = string.split("_")
    return liste[index]


def season(month):
    if (month == 12 or 1 <= month <= 4):
        return "winter"   
    elif (4 <= month <= 5):
        return "spring" 
    elif (6 <= month <= 9):
        return "summer"
    else:
        return "fall"

def wind_cord(v,o):
    
    a = o*math.pi/180

    return v*math.cos(a), v*math.sin(a)


def wind_vx_vy(df):
  Vx =  []
  Vy = []
  df_dict = df.to_dict('records')
  for line in tqdm(df_dict):
        o = line['wind_direction']
        v = line['wind_speed']
        vx,vy = wind_cord(v,o)
        Vx.append(vx)
        Vy.append(vy)
  df["Vx"] = Vx
  df["Vy"] = Vy
  return df


if __name__=='__main__':

    train = Read_Parquet(train_path)
    test = Read_Parquet(test_path)
    month_id = pd.read_csv(month_id_path)


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
    print(train.dtypes)

    train["Month"] = train["date"].dt.month.astype('int16')
    train["season"] = train['Month'].apply(season)

    test['season'] = test['Month'].apply(season)

    test = wind_vx_vy(test)
    train = wind_vx_vy(train)

    train.to_parquet(os.path.join(output_data_path ,"first_eng_final_imputed_train_x_station.gzip"), engine='fastparquet', compression='gzip')
    test.to_parquet(os.path.join(output_data_path ,"first_eng_final_imputed_test_x_station.gzip"), engine='fastparquet', compression='gzip')

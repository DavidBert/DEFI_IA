import pandas as pd
import os

path_test = '../input/Test/X_stations/X_station_test.csv'
path_train = '../input/Train/X_stations/X_station_train.csv'
coords_path = '../input/other/stations_coordinates.csv'
output_data = '../input/output_data'

def spliting(string, index):
    liste = string.split("_")
    return liste[index]


if __name__=='__main__':

    train = pd.read_csv(path_train)
    train.rename(columns={"ff":"wind_speed","t":"temp","hu":"humidity","dd":"wind_direction","td":"dew_point_temp"}, inplace=True)
    coords = pd.read_csv(coords_path)
    train = train.merge(coords, on=['number_sta'], how='left')
    train.height_sta = train.height_sta.astype('int16')
    test = pd.read_csv(path_test)
    test.rename(columns={"ff":"wind_speed","t":"temp","hu":"humidity","dd":"wind_direction","td":"dew_point_temp"}, inplace=True)
    test['number_sta'] = test['Id'].apply(lambda x : spliting(x, 0))
    test.number_sta = test.number_sta.astype(int)
    test = test.merge(coords, on=['number_sta'], how='left')
    test.height_sta = test.height_sta.astype('int16')
    test.to_parquet(os.path.join(output_data,"X_test_stations_prepared.gzip"), compression='gzip', engine='fastparquet')
    train.to_parquet(os.path.join(output_data,"X_train_stations_prepared.gzip"), compression='gzip', engine='fastparquet')



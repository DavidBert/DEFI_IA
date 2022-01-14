import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt

import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

import pre_traitement
import traitement_forecast



def treatement_train(df_X_train, coords, df_Y_train):
    #on supprime les heures de 'date'
    df_X_train["date"] = df_X_train["date"].apply(lambda x: dt.date(x.year, x.month, x.day))
    #ffill et bfill par station et par jour
    df_X_train = pre_traitement.fill_na_hour(df_X_train, 'date')
    #moyenne/somme pour avoir 1 valeur par jour et par station
    df_X_train = pre_traitement.hour_to_day(df = df_X_train, var = "date")
    #on ajoute la variable month
    df_X_train = pre_traitement.add_month(df_X_train)
    #on ajoute les coordonnées des stations
    df_X_train = pre_traitement.add_coords(coords, df_X_train)
    #on ajoute ground truth
    df_train = pre_traitement.merge_X_Y ( df_X_train, df_Y_train)
    #drop les nan de ground_truth
    df_train.dropna(subset = ["Ground_truth"], inplace = True)
    return df_train


def treatement_test(df_X_test, coords, baseline):
    #Création de l'Id 
    df_X_test["Id"] = df_X_test["Id"].apply(lambda x: x.split('_')[0]+'_'+x.split('_')[1])
    #Création de la variable 'day' et 'number_sta'
    df_X_test["date"] = df_X_test["Id"].apply(lambda x: x.split('_')[1]).astype('int')
    df_X_test["number_sta"] = df_X_test["Id"].apply(lambda x: x.split('_')[0])
    #ffill et bfill par station et par jour
    df_X_test = pre_traitement.fill_na_hour(df_X_test, 'date')
    #moyenne/somme pour avoir 1 valeur par jour et par station
    df_X_test = pre_traitement.hour_to_day(df = df_X_test, var = "date")
    #suppression des lignes qui ne sont pas dans la baseline
    df_X_test = df_X_test[df_X_test.Id.isin(baseline['Id'])] 
    #on ajoute la variable month
    df_X_test = pre_traitement.add_month(df_X_test)
    #on ajoute les coordonnées des stations
    df_X_test = pre_traitement.add_coords(coords, df_X_test)
    return df_X_test


def add_forecast(coords, df_train, df_X_test, name_forecast, data_path, var, K):
    # calcul des distances
    model = name_forecast
    path =  data_path + "/data_meteonet/train/X_forecast/" + name_forecast + "_train/"
    df_distance1 = traitement_forecast.calcul_distance(coords, path, model, K)
    
    
    #ajout des prévisions dans df_train
    p = 1 #paramètre d'interpolation
    df_train = df_train.sort_values(by=["date", "number_sta"]) #indispensable pour faire traitement_forecast
    df_train = traitement_forecast.add_prevision(p,df_distance1, df_train, path, model, var)
    
    #ajout des prévisions dans X_test
    path =  data_path + '/data_meteonet/test/' + name_forecast +"/"
    df_X_test = df_X_test.sort_values(by=["date", "number_sta"]) #indispensable pour faire traitement_forecast
    df_X_test = traitement_forecast.add_prevision(p,df_distance1,df_X_test, path, model, var = var,bool_train = False) 
    
    
    return df_train, df_X_test


def add_forecast3D(df_train, df_X_test, coords, data_path):
    model = 'arpege_3D_height'
    path =  data_path + '/data_meteonet/train/X_forecast/3D_arpege_train/'
    p = 1
    K = 3
    df_distance3 = traitement_forecast.calcul_distance(coords, path, model, K)
    
    df_train = traitement_forecast.add_prevision_3D(p,df_distance3, df_train, path, model, bool_train = True)
    
    path =  data_path + '/data_meteonet/test/3D_arpege/'
    df_X_test = traitement_forecast.add_prevision_3D(p,df_distance3, df_X_test, path, model, bool_train = False)
    
    return df_train, df_X_test


def preprocessing(data_path): 

    path_coords = data_path + '/data_station/Other/Other/'
    path_train =  data_path + '/data_station/Train/Train/'
    path_test = data_path + '/data_station/Test/Test/'
    path_baseline = data_path + '/data_station/Test/Test/Baselines/'
    print("Load data...")

    coords, df_X_train, df_X_test, df_Y_train, baseline = pre_traitement.load_datasets(path_coords, path_train, path_test, path_baseline)
    
    # treatement X_station
    print("X_station treatment...")
    df_train = treatement_train(df_X_train, coords, df_Y_train)
    df_X_test = treatement_test(df_X_test, coords, baseline)
    
    #treatement forecast data
    # liste des variables présentes dans forecast
    var = ["ws", "p3031", "u10", "v10", "t2m", "d2m", "r", "tp", "msl"]
    print("X_forecast treatment...")
    df_train, df_X_test = add_forecast(coords, df_train, df_X_test, "2D_arome", data_path, var, K=5)
    df_train, df_X_test = add_forecast(coords, df_train, df_X_test, "2D_arpege", data_path, var, K=3)
    df_train, df_X_test = add_forecast3D(df_train, df_X_test, coords, data_path)
    
    
    #missforest
    print("Begin of the missforest...")
    #pour train
    df_train2 = df_train.drop(["date", "Id", "number_sta"], axis = 1)
    imputer = MissForest()
    X_imputed = imputer.fit_transform(df_train2)
    
    df1 = pd.DataFrame(X_imputed, columns = df_train2.columns)
    df1["date"] = df_train["date"]
    df1["Id"] = df_train["Id"]
    df1["number_sta"] = df_train["number_sta"]
    df_train = df1
    
    
    #pour test
    df_test2 = df_X_test.drop(["date", "Id", "number_sta"], axis = 1)
    imputer2 = MissForest()
    X_imputed2 = imputer2.fit_transform(df_test2)
    
    df2 = pd.DataFrame(X_imputed2, columns = df_test2.columns)
    df2["date"] = df_X_test["date"]
    df2["Id"] = df_X_test["Id"]
    df2["number_sta"] = df_X_test["number_sta"]
    
    df_X_test = df2

    df_train.to_csv(data_path + "/df_train_preprocessed.csv", index = False)
    df_X_test.to_csv(data_path + "/df_test_preprocessed.csv", index = False)
    
    
    return df_train, df_X_test
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

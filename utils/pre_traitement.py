import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import xarray as xr


def load_datasets(path_coords, path_train, path_test, path_baseline ):
    # Chargement des coordonnées GPS des stations
    coords_path  = path_coords + 'stations_coordinates.csv'
    coords = pd.read_csv(coords_path)

    # Chargement des données météorologiques des stations (train)
    path = path_train + 'X_station_train.csv'
    df_X_train = pd.read_csv(path ,parse_dates=['date'],infer_datetime_format=True)

    # Chargement des données des stations (test)
    path = path_test + 'X_station_test.csv'
    df_X_test = pd.read_csv(path)

    # Open Y_train
    fname = path_train + 'Y_train.csv'
    df_Y_train = pd.read_csv(fname, parse_dates=['date'], infer_datetime_format=True)

    # Chargement de la baseline
    baseline_path  = path_baseline + 'Baseline_observation_test.csv'
    baseline_obs_test = pd.read_csv(baseline_path)

    return coords, df_X_train, df_X_test, df_Y_train, baseline_obs_test

def merge_X_Y ( X_train, Y_train):
    data = Y_train
    df_train = X_train

    data["date"] = data["date"].apply(lambda x: x- dt.timedelta(days=1)) #on enlève un jour pour faire matcher le Y n+1 avec le X n
    df_train['date'] = df_train['date'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))

    df_train = df_train.merge(data[["date", "number_sta", "Ground_truth"]], on = ["date", "number_sta"], how = "left")
    
    return df_train

def add_month (df) : 
    
    if not "month" in df.columns :
        df["month"] = df["date"].apply(lambda x: x.month)

    # Changement du type de la variable month en facteur 
    df['month'] = pd.Categorical(df['month'], ordered=False)

    # Pour les variables qualitatives, générer les indicatrices
    # On convertit la variable non-numérique en numérique pour pouvoir calculer des distances entre observations/individus.
    # On transforme une variable qualitative à n modalités en n indicatrices correspondant à n nouvelles variables.
    df_Dum = pd.get_dummies(df[["month"]])

    # Enlever une modalité par variable qualitative pour avoir un modèle identifiable.
    del df_Dum["month_1"] 

    # Variables explicatives quantitatives
    df_Quant= df.drop('month', axis=1, inplace=False)

    # Variables explicatives
    df = pd.concat([df_Quant, df_Dum,],axis=1)
    
    return df

def hour_to_day(df, var):    
    
    aux = list(np.array(df.columns))
    aux.remove("ff")
    aux.remove("t")
    aux.remove("td")
    aux.remove("hu")
    aux.remove("dd")
    aux.remove("precip")
        
    # Moyenne des variables groupées par la date et la station
    sub_df1 = df[[var, 'number_sta',"ff", "t", "td", "hu", "dd"]].groupby([var, 'number_sta']).mean().reset_index()

    # Somme des précipitations sur la date et la station
    sub_df2 = df[[var, 'number_sta',"precip"]].groupby([var, 'number_sta']).sum().reset_index()

    # Récupération des autres variables 
    sub_df3 = df[aux].drop_duplicates([var, 'number_sta'])

    # Fusion des 3 sub_sets
    df = sub_df1.merge(sub_df2, on = [var, "number_sta" ], how = "left")
    df = df.merge(sub_df3, on = [var, "number_sta" ], how = "left")
    df.reset_index()
    
    return df

def fill_na_hour(df, var):

    df = df.groupby(['number_sta', var], sort=False).apply(lambda x: x.ffill().bfill())
    
    return df
    
    
def add_coords(coords, df):
    coords['number_sta'] = coords['number_sta'].astype('category')
    df['number_sta'] = df['number_sta'].astype("int")
    
    # fusion de df_X_train avec les coordonnées géographiques 
    df = df.merge(coords, on=['number_sta'], how='left')

    return df

# On clip les outliers
def find_outliers(series, q = 1.96):
    positive_outliers = (series[series>0] - series.mean()) >  q * series.std()
    negative_outliers = (series[series<0] - series.mean()) < -q * series.std()
    min_treshold = series[series<0][~negative_outliers].min()
    max_treshold = series[series>0][~positive_outliers].max()
    return min_treshold, max_treshold

    
    

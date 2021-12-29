import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler,RobustScaler

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Downloading dataset :
# import kaggle
# !kaggle competitions download -c defi-ia-2022

def X_train_preprocessing(df):
    
    # Récupération mois/jour/heure par ligne
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    hour = df["Id"].str.split("_", n = 2, expand = True)[2]
    df['hour'] = hour.astype(int)
    day = df["Id"].str.split("_", n = 2, expand = True)[1]
    df['day'] = day.astype(int)
    
    # Création d'un Id quotidien pour pouvoir merge X_train avec Y_train
    df['Id_merge'] = df['number_sta'].astype(str).str.cat(day,sep="_")
    
    # Mise en ordre des features
    df = df[['dd','hu','td','t','ff','precip','month','Id','Id_merge','number_sta','hour','day']]
    df['precip'] = df['precip']*24
    df['month'] = df['month'].astype(int)
    
    # Tri des features : N° Station / Jour / Heure
    df = df.sort_values(["number_sta","day",'hour'])
    df = df.drop(['hour','day'],axis=1)
    
    return df
                 
def Y_preprocessing(df):
    
    df = df.drop(['date','number_sta'],axis=1)
    df = df[['Id','Ground_truth']]
    df['Id_merge'] = df['Id']
    df = df.dropna()
    
    return df

def X_test_preprocessing(df):
    
    # Récupération N° Station/jour/heure par ligne
    hour = df["Id"].str.split("_", n = 2, expand = True)[2]
    df['hour'] = hour.astype(int)
    day = df["Id"].str.split("_", n = 2, expand = True)[1]
    df['day'] = day.astype(int)
    nb_station = df["Id"].str.split("_", n = 2, expand = True)[0]
    df['number_sta'] = nb_station.astype(int)
    
    # Tri des features : N° Station / Jour / Heure
    df = df.sort_values(["number_sta","day",'hour'])
    df['Id'] = df['number_sta'].astype(str).str.cat(day,sep="_")
    df = df.drop(['hour','day'],axis=1) 
    
    return df

def imputation(df):
    
    # Version 1 : DropNaNs
    # df = df.dropna()
    
    # Version 2 : IterativeImputer
    temp = df[["Id","number_sta","month"]]
    imp_mean = IterativeImputer(random_state=0)
    df = pd.DataFrame(imp_mean.fit_transform(df[["ff","t","td","hu","dd","precip","lat","lon","height_sta"]]))
    df = pd.concat([temp,df],axis=1)
    df.columns = ["Id","number_sta","month","ff","t","td","hu","dd","precip","lat","lon","height_sta"]

    # Version 3 : KNNImputer
    # temp = df[["Id","number_sta","month"]]
    # imputer = KNNImputer(n_neighbors=2)
    # df = pd.DataFrame(imputer.fit_transform(df[["ff","t","td","hu","dd","precip","ws","lat","lon","height_sta"]]))
    # df = pd.concat([temp,df],axis=1)
    # df.columns = ["Id","number_sta","month","ff","t","td","hu","dd","precip","ws","lat","lon","height_sta"]
    
    return df

# Importing the data sets
X_train_df = pd.read_csv('./defi-ia-2022/Train/Train/X_station_train.csv')
X_train_df = X_train_preprocessing(X_train_df)

Y_train_df = pd.read_csv('./defi-ia-2022/Train/Train/Y_train.csv')
Y_train_df = Y_preprocessing(Y_train_df)

X_test_df = pd.read_csv('./defi-ia-2022/Test/Test/X_station_test.csv')
X_test_df = X_test_preprocessing(X_test_df)

Baseline = pd.DataFrame(pd.read_csv('./defi-ia-2022/Test/Test/Baselines/Baseline_observation_test.csv')['Id'])

coords = pd.read_csv('./defi-ia-2022/Other/Other/stations_coordinates.csv')

# Creating trainset
trainset = X_train_df.merge(Y_train_df,how="inner",on="Id_merge")
trainset['Id'] = trainset['Id_merge']
trainset = trainset.drop(['Id_x','Id_merge','Id_y'],axis=1)
trainset = trainset.groupby("Id").mean()
trainset = trainset.reset_index()
trainset['month'] = trainset['month'].astype(int)
trainset = trainset.merge(coords,how="inner",on="number_sta")

# Creating testset
testset = X_test_df.merge(coords,how="left",on="number_sta")

# Imputation
trainset = imputation(trainset)
testset = imputation(testset)
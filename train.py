print("--- Importing Libraries ---")
# Usuals
import numpy as np
import pandas as pd

# SKLearn
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Filter pandas warnings
import warnings
warnings.filterwarnings('ignore')

# Modeling
import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dropout, Dense

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

def train_imputation(df):
    
    # Version 1 : DropNaNs
    # df = df.dropna()
    
    # Version 2 : IterativeImputer
    temp = df[["Id","number_sta","month","Ground_truth"]]
    imp_mean = IterativeImputer(random_state=0)
    df = pd.DataFrame(imp_mean.fit_transform(df[["ff","t","td","hu","dd","precip","lat","lon","height_sta"]]))
    df = pd.concat([temp,df],axis=1)
    df.columns = ["Id","number_sta","month","Ground_truth","ff","t","td","hu","dd","precip","lat","lon","height_sta"]

    # Version 3 : KNNImputer
    # temp = df[["Id","number_sta","month"]]
    # imputer = KNNImputer(n_neighbors=2)
    # df = pd.DataFrame(imputer.fit_transform(df[["ff","t","td","hu","dd","precip","ws","lat","lon","height_sta"]]))
    # df = pd.concat([temp,df],axis=1)
    # df.columns = ["Id","number_sta","month","ff","t","td","hu","dd","precip","ws","lat","lon","height_sta"]
    
    return df

def test_imputation(df):
    
    # Version 2 : IterativeImputer
    temp = df[["Id","number_sta","month"]]
    imp_mean = IterativeImputer(random_state=0)
    df = pd.DataFrame(imp_mean.fit_transform(df[["ff","t","td","hu","dd","precip","lat","lon","height_sta"]]))
    df = pd.concat([temp,df],axis=1)
    df.columns = ["Id","number_sta","month","ff","t","td","hu","dd","precip","lat","lon","height_sta"]
    
    return df

print("--- Loading Data ---","\n")
        
# Importing the data sets
X_train_df = pd.read_csv('./defi-ia-2022/Train/Train/X_station_train.csv')
X_train_df = X_train_preprocessing(X_train_df)
print("        - X_train Loaded -")
print("          Shape : ",X_train_df.shape,"\n")

Y_train_df = pd.read_csv('./defi-ia-2022/Train/Train/Y_train.csv')
Y_train_df = Y_preprocessing(Y_train_df)
print("        - Y_train Loaded -")
print("          Shape : ",Y_train_df.shape,"\n")

X_test_df = pd.read_csv('./defi-ia-2022/Test/Test/X_station_test.csv')
X_test_df = X_test_preprocessing(X_test_df)
print("        - X_test Loaded -")
print("          Shape : ",X_test_df.shape,"\n")

Baseline = pd.DataFrame(pd.read_csv('./defi-ia-2022/Test/Test/Baselines/Baseline_observation_test.csv')['Id'])
print("        - Baseline Loaded -")
print("          Shape : ",Baseline.shape,"\n")

coords = pd.read_csv('./defi-ia-2022/Other/Other/stations_coordinates.csv')
print("        - Coordinates Loaded -")
print("          Shape : ",coords.shape,"\n")

# Creating trainset
print("--- Preprocessing Data ---","\n")
trainset = X_train_df.merge(Y_train_df,how="inner",on="Id_merge")
trainset['Id'] = trainset['Id_merge']
trainset = trainset.drop(['Id_x','Id_merge','Id_y'],axis=1)
trainset = trainset.groupby("Id").mean()
trainset = trainset.reset_index()
trainset['month'] = trainset['month'].astype(int)
trainset = trainset.merge(coords,how="inner",on="number_sta")
# Imputation
trainset = train_imputation(trainset)
print("        - Trainset Created -")
print("          Shape : ",trainset.shape)
print("          Columns : ",trainset.columns,"\n")

# Creating testset
testset = X_test_df.merge(coords,how="left",on="number_sta")
# Imputation
testset = test_imputation(testset)
print("        - Testset Created -")
print("          Shape : ",testset.shape)
print("          Columns : ",testset.columns,"\n")

print("--- Data Preprocessed ---","\n")

# Train Test Split

print("--- Splitting data for training ... ---")
from sklearn.model_selection import train_test_split
trainset, valset = train_test_split(trainset, test_size=0.3)
X_train = trainset.drop(['Ground_truth','Id'],axis=1)
y_train = trainset['Ground_truth']
X_test = valset.drop(['Ground_truth','Id'],axis=1)
y_test = valset['Ground_truth']
print("--- Splitted in trainset/valset ---")

# Neural network
ann = Sequential()
ann.add(BatchNormalization())
ann.add(Dense(350, activation="relu", kernel_initializer='normal'))
ann.add(Dropout(0.3))
ann.add(Dense(512, activation="relu", kernel_initializer='normal'))
ann.add(Dropout(0.3))
ann.add(Dense(512, activation="relu", kernel_initializer='normal'))
ann.add(Dropout(0.3))
ann.add(Dense(128, activation="relu", kernel_initializer='normal'))
ann.add(Dense(1))

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2,
#     decay_steps=10000,
#     decay_rate=0.9)
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

###
keras.backend.set_epsilon(1)
# Solves problem of really high metric :
# https://stackoverflow.com/questions/49729522/why-is-the-mean-average-percentage-errormape-extremely-high

ann.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])

history = ann.fit(X_train,y_train,epochs=2048, batch_size=8192, verbose = 2 ,validation_data=(X_test,y_test))
print("--- Model Trained ---")
history_dict = history.history

ann.save("Modele_6.h5")
print("--- Model Saved ---")

print("Neural Network train MAPE :",round(history_dict['mean_absolute_percentage_error'][-1],2))
print("Neural Network validation MAPE :",round(history_dict['val_mean_absolute_percentage_error'][-1],2))
print("--- Importing Libraries ---")
# Usuals
import numpy as np
import pandas as pd
import time
import random
import warnings
warnings.filterwarnings('ignore')

# SKLearn
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Modeling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler,MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


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

# print("--- Splitting data for training ... ---")
# trainset, valset = train_test_split(trainset, test_size=0.3)
# X_train = trainset.drop(['Ground_truth','Id'],axis=1)
# y_train = trainset['Ground_truth']
# X_test = valset.drop(['Ground_truth','Id'],axis=1)
# y_test = valset['Ground_truth']
# print("--- Splitted in trainset/valset ---")

# Modeling
reg = KNeighborsRegressor(n_neighbors=3)

def compute_mape(model):
    y_pred_temp = model.predict(x_test) + 1
    y_test_temp = y_test + 1
    temp = np.abs(y_pred_temp-y_test_temp)/y_test_temp
    MAPE = (100/len(temp))*np.sum(temp)
    return MAPE

test_size = 0.2  #Rapport de division
N_trials = 10  #Nombre d'essais
mapes= []
for i in range(N_trials):
    print(f"Trial {i+1}")
    random_state = random.randint(0, 1000)
    trainset, testset = train_test_split(trainset, test_size=test_size, random_state=random_state)
    x_train = trainset.drop(['Ground_truth','Id'],axis=1)
    y_train = trainset['Ground_truth']
    x_test = testset.drop(['Ground_truth','Id'],axis=1)
    y_test = testset['Ground_truth']
    print("Modèle : KNeighborsRegressor")
    reg = make_pipeline(StandardScaler(),reg)
    start = time.time()
    reg.fit(x_train,y_train)
    print("Time :",round(time.time()-start,3),"s")
    mape = compute_mape(reg)  #Calcul MAPE
    mapes.append(mape)  #Stockage

print("MAPE Dictionnary :",mapes)
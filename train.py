import sys, argparse, warnings, time, random
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_folder')
    parser.add_argument('-d', '--data_path')
    args = parser.parse_args()
    output_folder = "./"+args.output_folder
    input_folder = "./"+args.data_path


print("--- Importing Libraries ---")

# Usuals
import numpy as np
import pandas as pd
import pickle

# Modeling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import Modules
sys.path
sys.path.append('./modules')
from test_imputation import *
from train_imputation import *
from X_test_feature_engineering import *
from X_train_feature_engineering import *
from Y_feature_engineering import *
from Trainset_creation import *




print("--- Loading Data ---","\n")
        
# Importing the data sets
X_train_df = pd.read_csv(input_folder+'/Train/Train/X_station_train.csv')
X_train_df = X_train_preprocessing(X_train_df)
print("        - X_train Loaded -")
print("          Shape : ",X_train_df.shape,"\n")

Y_train_df = pd.read_csv(input_folder+'/Train/Train/Y_train.csv')
Y_train_df = Y_preprocessing(Y_train_df)
print("        - Y_train Loaded -")
print("          Shape : ",Y_train_df.shape,"\n")

X_test_df = pd.read_csv(input_folder+'/Test/Test/X_station_test.csv')
X_test_df = X_test_preprocessing(X_test_df)
print("        - X_test Loaded -")
print("          Shape : ",X_test_df.shape,"\n")

Baseline = pd.DataFrame(pd.read_csv(input_folder+'/Test/Test/Baselines/Baseline_observation_test.csv')['Id'])
print("        - Baseline Loaded -")
print("          Shape : ",Baseline.shape,"\n")

coords = pd.read_csv(input_folder+'/Other/Other/stations_coordinates.csv')
print("        - Coordinates Loaded -")
print("          Shape : ",coords.shape,"\n")

# Creating trainset
print("--- Preprocessing Data ---","\n")
trainset = Trainset_creation(X_train_df,Y_train_df)
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



test_size = 0.2  #Rapport de division
N_trials = 10  #Nombre d'essais
mapes= []
start = time.time()

print("--- Training",N_trials,"Folds ---","\n")
for i in range(N_trials):
    
    print("     Fold",i," ---","\n")
    
    # Création d'autant de datasets qu'il y a de folds
    random_state = random.randint(0, 1000)
    train, validation = train_test_split(trainset, test_size=test_size, random_state=random_state)
    x_train = train.drop(['Ground_truth','Id'],axis=1)
    y_train = train['Ground_truth']
    x_test = validation.drop(['Ground_truth','Id'],axis=1)
    y_test = validation['Ground_truth']
    
    # Entrainement du modèle
    # Modeling
    reg = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    reg = make_pipeline(StandardScaler(),reg)
    reg.fit(x_train,y_train)
    
    # Prediction
    y_pred_temp = reg.predict(x_test) + 1
    y_test_temp = y_test + 1
    temp = np.abs(y_pred_temp-y_test_temp)/y_test_temp
    MAPE = (100/len(temp))*np.sum(temp)
    mapes.append(MAPE)  #Stockage
    
print("Training",N_trials,"folds took :",round(time.time()-start,3),"s")

print("Mean Training MAPE on validation set :",round(np.mean(mapes),2))

print("--- Making Predictions and Saving Model ---","\n")
# save the model to disk
filename = './Results/model.sav'
pickle.dump(reg, open(filename, 'wb'))
y_pred = reg.predict(testset.drop("Id",axis=1))
y_pred.to_csv('./Results/predictions.csv',index=True)
print("        - Prediction saved -")
print("          Shape : ",y_pred.shape)
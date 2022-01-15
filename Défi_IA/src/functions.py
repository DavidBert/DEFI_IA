import pandas as pd
import numpy as np
import sklearn
from tqdm import tqdm 
import math
import os

# modelization 
import sklearn
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

from scipy.stats import kurtosis
from scipy.stats import skew

## Hyperparametres tuning
import optuna 
from optuna import Trial, visualization
from optuna.samplers import TPESampler

## Model saving ( serialization)
import joblib


from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb


### Visualizations 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import sweetviz as sv
import ppscore as pps
import datetime as d
import missingno as msno




test_path_input = '../input/output_data/X_test_stations_prepared.gzip'
train_path_input = '../input/output_data/X_train_stations_prepared.gzip'
output_viz = '../input/output_viz'
y_train_path = '../input/Train/X_stations/Y_train.csv'



def dist_feature(dataframe, feature, folder=output_viz):
    plt.figure(figsize=(20,10))
    ax = sns.distplot(x=dataframe[feature], label = feature)
    plt.title(f"{feature}'s distribution training data")
    plt.xlabel(f"{feature}")
    ax.grid(True)
    plt.savefig(os.path.join(output_viz, f"{feature}'s_distribution_training_data.png"))
    # plt.show()

def paiplot(df):
    # paiplot to see the visually the relations between the features
    plt.figure(figsize=(20,15))
    sns.pairplot(df.drop(['date','number_sta','Id'], axis=1))# , diag_kind="kde"
    plt.title("Pairplot of training data")
    plt.savefig(os.path.join(output_viz,"Pairplot_train_features.png"))
    # plt.show()


def linear_correlation(train):
    plt.figure(figsize=(30,15))  
    dataplot = sns.heatmap(train.corr(), cmap="YlGnBu", annot=True)
    plt.title("Correlation matrix of all the feature")
    plt.savefig(os.path.join(output_viz,"Correlation_matrix_between_features.png"))
    # plt.show()

def non_linear_correlation(train):
    plt.figure(figsize=(30,15))
    matrix_df = pps.matrix(train.drop(['number_sta','date','Id'], axis=1))[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
    sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
    plt.title('Ppscore metric of all the features on training data')
    plt.savefig(os.path.join(output_viz,"ppscore_train_stations.png"))
    # plt.show()

def boxplot(dataframe, x, folder=output_viz):
    plt.figure(figsize=(20,8))
    ax = sns.boxplot(x=x,data=dataframe, dodge=False)
    plt.title(f"Boxplot of feature {x} for training data ")
    plt.ylabel(f"{x}")
    ax.grid(True)
    plt.savefig(os.path.join(output_viz,f"Boxplot_of_feature_{x}.png"))
    # plt.show()

def msno_bar(train):
    msno.bar(train)
    plt.savefig(os.path.join(output_viz,f"Missing_values_data_Distribution.png"))


def msno_matrix(train):
    msno.matrix(train)
    plt.savefig(os.path.join(output_viz,f"Missing_values_data_nullity matrix.png"))


def msno_dendrogram(train):
    msno.dendrogram(train)
    plt.savefig(os.path.join(output_viz,f"Missing_values__data_endrogram.png"))


def msno_heatmap(train):
    msno.heatmap(train, cmap='YlGnBu')
    plt.savefig(os.path.join(output_viz,f"Missing_values_data_heatmap.png"))

def spliting(string, index):
    liste = string.split("_")
    return liste[index]

def Read_Parquet(chemin):
      return pd.read_parquet(chemin, engine='fastparquet')


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


def training_kfold(df, kfold): ## stratified
    df.loc[:,"kfold"] = -1
    KfoldsDf_train = df.reset_index().sample(frac=1)
    kf = model_selection.KFold(n_splits=kfold)
    for f, (t_, v_) in enumerate(kf.split(X=KfoldsDf_train)): 
        KfoldsDf_train.loc[v_, 'kfold'] = f 
    return KfoldsDf_train


def run(KfoldsDf_train, fold, model, metric):
    
    y_train = KfoldsDf_train[KfoldsDf_train["kfold"]!=fold]['Ground_truth'].values
    y_val = KfoldsDf_train[KfoldsDf_train["kfold"]==fold]['Ground_truth'].values
    X_train = KfoldsDf_train[KfoldsDf_train["kfold"]!=fold].drop(['kfold',"Ground_truth"], axis=1).set_index('Id')
    X_val =KfoldsDf_train[KfoldsDf_train["kfold"]==fold].drop(['kfold',"Ground_truth"], axis=1).set_index('Id')

    model.fit(X_train,y_train)
 
    valid_preds = model.predict(X_val) 
    valid_preds += 1
    y_val +=1
    score = metric(y_val, valid_preds) 
    model_name = type(model).__name__
    
    return score, model_name, fold





def objective(trial: Trial,X,y) -> float:
    
    joblib.dump(study, 'study.pkl')

    param = {
                "n_estimators" : trial.suggest_int('n_estimators', 0, 1000),
                'max_depth':trial.suggest_int('max_depth', 2, 25),
                'reg_alpha':trial.suggest_int('reg_alpha', 0, 5),
                'reg_lambda':trial.suggest_int('reg_lambda', 0, 5),
                'min_child_weight':trial.suggest_int('min_child_weight', 0, 5),
                'gamma':trial.suggest_int('gamma', 0, 5),
                'learning_rate':trial.suggest_loguniform('learning_rate',0.005,0.5),
                'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.1,1,0.01),
                'n_jobs' : -1,
                'objective':'reg:pseudohubererror',
                'gpu_id':0,
            }
    
    model = xgb.XGBRegressor(**param)
    
    return cross_val_score(model, X, y, cv=10).mean()
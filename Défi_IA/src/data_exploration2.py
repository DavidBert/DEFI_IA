import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import os
import sweetviz as sv
import ppscore as pps

from functions import Read_Parquet


folder_result = "../input/output_viz/"


train_path = "../input/imputed_data/X_Train_concat_y_train.gzip"

test_path = "../input/imputed_data/X_Test_merged_clean.gzip"
def Read_Parquet(chemin):
      return pd.read_parquet(chemin, engine='fastparquet')

def dist_feature(dataframe, feature):
    
    plt.figure(figsize=(20,10))
    
    ax = sns.distplot(x=dataframe[feature], label = feature)

    plt.title(f"{feature}'s distribution in concat train data with target")

    plt.xlabel(f"{feature}")
    ax.grid(True)
    plt.savefig(os.path.join(folder_result,f"{feature}'s_distribution_new_concat_training_data.png"))

def boxplot(dataframe, x):
    plt.figure(figsize=(20,8))
    ax = sns.boxplot(x=x,data=dataframe, dodge=False)
    plt.title(f"Boxplot of feature {x} for training data ")
    plt.ylabel(f"{x}")
    ax.grid(True)
    plt.savefig(os.path.join(folder_result,f"Boxplot_of_feature_{x}_with_target__training_data.png"))


def pairplot(df, feature):
  plt.figure(figsize=(20,15))
  sns.pairplot(train[[feature, 'Ground_truth']],diag_kind="kde")# , 
  plt.title(f"Pairplot of {feature} Ground truth")
  plt.savefig(os.path.join(folder_result,f"Pairplot of {feature} Ground truth.png"))


if __name__=='__main__':
      train = Read_Parquet(train_path)
      test = Read_Parquet(test_path)
      my_report  = sv.analyze([train,'Train'], target_feat='Ground_truth')
      my_report  = sv.analyze([train,'Train'])
      my_report.show_html(os.path.join(folder_result,'First_train_report.html'))
      my_report  = sv.analyze([test,'Test'])
      my_report.show_html(os.path.join(folder_result,'First_test_report.html'))


      numerical_features = ['temp_min', 'temp_max', 'temp_mean', 'temp_median',
       'temp_percentile_25', 'temp_percentile_75', 'temp_skew',
       'temp_kurtosis', 'humidity_min', 'humidity_max', 'humidity_mean',
       'humidity_median', 'humidity_percentile_25', 'humidity_percentile_75',
       'humidity_skew', 'humidity_kurtosis', 'dew_point_temp_min',
       'dew_point_temp_max', 'dew_point_temp_mean', 'dew_point_temp_median',
       'dew_point_temp_percentile_25', 'dew_point_temp_percentile_75',
       'dew_point_temp_skew', 'dew_point_temp_kurtosis', 'Vx_min', 'Vx_max',
       'Vx_mean', 'Vx_median', 'Vx_percentile_25', 'Vx_percentile_75',
       'Vx_skew', 'Vx_kurtosis', 'Vy_min', 'Vy_max', 'Vy_mean', 'Vy_median',
       'Vy_percentile_25', 'Vy_percentile_75', 'Vy_skew', 'Vy_kurtosis',
       'precip_sum', 'precip_min', 'precip_max', 'precip_mean',
       'precip_median', 'precip_percentile_25', 'precip_percentile_75',
       'precip_skew', 'precip_kurtosis', 'lat', 'lon',
       'height_sta', 'Ground_truth']

      for feat in numerical_features:
            dist_feature(train, feat)
      for feat in numerical_features:
            boxplot(train, feat)
      train['Month'] = train['Month'].astype('category').cat.codes
      train['season'] = train['season'].astype('category').cat.codes

      plt.figure(figsize=(30,15))  
      dataplot = sns.heatmap(train.corr(), cmap="YlGnBu", annot=True)
      plt.title("Correlation matrix of all the feature with target")
      plt.savefig(folder_result+ "Correlation_matrix_between_features_&_target.png")


      pairplot_features = ['temp_min', 'temp_max', 'temp_mean', 'temp_median',
       'temp_percentile_25', 'temp_percentile_75', 'temp_skew',
       'temp_kurtosis', 'humidity_min', 'humidity_max', 'humidity_mean',
       'humidity_median', 'humidity_percentile_25', 'humidity_percentile_75',
       'humidity_skew', 'humidity_kurtosis', 'dew_point_temp_min',
       'dew_point_temp_max', 'dew_point_temp_mean', 'dew_point_temp_median',
       'dew_point_temp_percentile_25', 'dew_point_temp_percentile_75',
       'dew_point_temp_skew', 'dew_point_temp_kurtosis', 'Vx_min', 'Vx_max',
       'Vx_mean', 'Vx_median', 'Vx_percentile_25', 'Vx_percentile_75',
       'Vx_skew', 'Vx_kurtosis', 'Vy_min', 'Vy_max', 'Vy_mean', 'Vy_median',
       'Vy_percentile_25', 'Vy_percentile_75', 'Vy_skew', 'Vy_kurtosis',
       'precip_sum', 'precip_min', 'precip_max', 'precip_mean',
       'precip_median', 'precip_percentile_25', 'precip_percentile_75',
       'precip_skew', 'precip_kurtosis', 'Month', 'season']

      for feat in pairplot_features:
        pairplot(train, feat)

      
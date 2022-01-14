# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:42:38 2022
@author: Noemi
"""
import numpy as np
import pandas as pd
import utils.feature_engineering as fe
import utils.MLP_model as MLP
import utils.LGBM_model as LGBM
import sys 
import os
import argparse

    

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_path', help='Folder to data files')
parser.add_argument(
    '--output_folder', help='Folder to save the model, training logs, and predictions',
)
args = parser.parse_args()

data_path = args.data_path
results_path = args.output_folder

for path in [data_path, results_path]:
    if not os.path.exists(path):
        os.makedirs(path)


# Load train files
X_train_station = os.path.join(data_path,'Train/Train/full_X_train.csv')
Y_train_station = pd.read_csv(os.path.join(data_path,'Train/Train/full_Y_train.csv'),sep=",",header=0)

# Load test files
path_baseline_test = os.path.join(data_path,'Test/Test/Baselines/')
X_station_test = os.path.join(data_path,"Test/Test/full_X_test.csv")

########################################################################
#                                                                      #
#                         Feature Engineering                          #
#                                                                      #
########################################################################

    



# Train set
X_station_train_mean=fe.train_feature(X_train_station)
# Test set 
X_station_test_mean = fe.test_feature(X_station_test)


MLP.modele(X_station_train_mean,Y_train_station,X_station_test_mean)
LGBM.modele(X_station_train_mean,Y_train_station,X_station_test_mean)
print("predictions .csv files were saved in results folder.")



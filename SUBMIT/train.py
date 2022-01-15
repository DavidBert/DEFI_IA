import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
import modules_fp as modu
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime
import joblib
import pickle
import argparse

# parse path data and results
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str)
parser.add_argument('--output_folder',type=str)
args = parser.parse_args()
path_data = args.data_path
path_results = args.output_folder 


################################################################################
#                   MAIN PROGRAM                                               #
################################################################################
# GLOBAL PARAMETERS
epochs=150
id_experiment = 'e'+str(epochs)+'_with_all'
id_experiment = ''
# name train csv
fname_train =   path_data + '/Train/Train/X_station_train.csv'

# DATES definition
first_date= datetime.datetime(2016,1,1)
#last_date = datetime.datetime(2017,12,30)
last_date = datetime.datetime(2017,12,30)

# Select stations with data for all the days and less than 25 percent of Nans 
# then save the selected stations ids in results folder 
activate_select_stations=True

if activate_select_stations:
  selected_stations = modu.select_stations(first_date, last_date, fname_train)
  np.save(path_results+'/selected_stations', selected_stations)

################################################################################
#     OPEN SELECTED STATIONS AND CREATE DATABASE FOR TRAINING                  #
################################################################################

# Open Selected Stations Ids
# ------------------------------------------------------------------------------
selected_sta = np.load(path_results+'/selected_stations.npy')

# Create dataset
# ------------------------------------------------------------------------------
if True:
  print('Create and save dataset...')
  modu.create_and_save_dataset(fname_train, selected_sta, path_results, first_date, last_date)

# Load dataset
# ------------------------------------------------------------------------------
if True:
  print('Load dataset...')
  X_train, y_train, X_val , y_val = modu.load_dataset(path_results)

# Model
# ------------------------------------------------------------------------------
if True:
  print('Model...')
  model = modu.create_model(X_train.shape[1], mod_ = 'regr')

# Train
# ------------------------------------------------------------------------------
if True:
  print('Train...')
  history_to_save = model.fit(X_train, y_train, epochs=epochs, batch_size=1000)
  
# Save model
# ------------------------------------------------------------------------------
if True:
  print('Save model...')
  model_name = '/my_model'+id_experiment+'.h5'
  model.save(path_results + model_name)
  history_file = model_name + '_history'+id_experiment
  with open(path_results + history_file, 'wb') as file_pi:
      pickle.dump(history_to_save.history, file_pi)


# Load model
# ------------------------------------------------------------------------------
if True:
  print('Load model...')
  model_name = '/my_model'+id_experiment+'.h5'
  model = load_model(path_results + model_name)
  history_file = model_name + '_history'+id_experiment
  history = pickle.load(open(path_results+history_file, "rb"))

# Validate
# ------------------------------------------------------------------------------
if True:
  print('Validate...')
  prediction = model.predict(X_val, batch_size=1000)
  prediction = np.maximum(prediction, np.ones_like(prediction))
  mape_val = modu.mape(y_val,prediction)
  print('MAPE:',modu.mape(y_val,prediction))
  l_file = path_results + '/log_file.txt'
  txt_log = 'Id_exp:'+id_experiment+', MAPE:'+str(mape_val)+', Loss mean at the end of training:'+str(np.mean(history['loss'][-5:]))+'\n'
  modu.write_log(l_file,txt_log)
  
################################################################################
#  PREDICTION TEST                                                             #
################################################################################
if True:
  print('Create and save test set...')
  X_test, index_id = modu.create_and_save_test_set(path_data, path_results, path_results + "/scaler.save")

  # select only the index present in the baselines 
  fobs = path_data + '/Test/Test/Baselines/Baseline_observation_test.csv'
  i_to_predict = modu.idx2predict(index_id, fobs)

  # Adjust X_test
  X_test_sel = X_test[i_to_predict]
  index_id_sel = index_id[i_to_predict]

# Predict
if True:
  model_name = '/my_model.h5'
  model = load_model(path_results + model_name)

if True:
  prediction = model.predict(X_test_sel, batch_size=100)
  # subtitute by one if prediction is inferior 
  prediction = np.maximum(prediction, np.ones_like(prediction))

# Save csv
if True:
  name_csv=path_results+'/predictions.csv'
  d = {'Id': index_id_sel.squeeze(), 'Prediction': prediction.squeeze()}
  df_res = pd.DataFrame(data=d)
  df_res.to_csv(name_csv, index=False)































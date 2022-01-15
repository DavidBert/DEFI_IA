###################################### Librairies à charger

import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
import xarray as xr
import os
import time
import argparse
import pandas as pd 
import gc
from datetime import date
from numpy import genfromtxt
import csv
#import tensorflow.keras as keras
#from keras.models import Model, load_model
#from keras.layers import Input,ConvLSTM2D,TimeDistributed,Concatenate,Add,Bidirectional,Concatenate, dot, add, multiply, Activation, Reshape, Dense, RepeatVector, Dropout, Permute, LSTM, Dense, Flatten, Embedding
#from keras.layers.convolutional import Conv3D,Conv2D,SeparableConv2D, Cropping2D, Cropping3D,Conv2DTranspose, UpSampling2D, DepthwiseConv2D, MaxPooling2D, Conv1D, MaxPooling1D
#from keras.layers import Lambda
#import tensorflow as tf
###################################### Fichiers *.py à importer
from fonctions import *
######################################


###################################### Main Program


if __name__=='__main__':
    ts = time.time()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default = '', help='votre chemin perso du dossier contenant les fichiers téléchargés')
    parser.add_argument('--output_folder', type=str, default = '', help='votre chemin perso du dossier output')
    
    # Récupération des chemins d'accès des dossiers locaux mis en argument dans la commande d'exécution du train.py
    args = parser.parse_args()
    data_path = args.data_path
    output_folder = args.output_folder
    
    
    # Importations des coordonnées et dates
    print("----- Importations des coordonnées et dates -----")
    data = genfromtxt(data_path+'/stations_coordinates.csv', delimiter=',', skip_header = 1)
    coordonnees_stations = data[:,1:3]
    numéros_stations = data[:,0]
    altitude_stations = data[:,-1]

    journées_train = pd.date_range(start="2016-01-01",end="2017-12-31")
    liste_journées_train = np.array(str(journées_train[0])[:10],dtype =str)
    for i in range(1,len(journées_train)):
        liste_journées_train = np.append(liste_journées_train,str(journées_train[i])[:10])
    journées_train = liste_journées_train
    journées_test = [i for i in range(363)]
    id_month_test = genfromtxt(data_path+'/Id_month_test.csv', delimiter=',', skip_header = 1)
    
    # Génération des fichiers csv
    print("----- Génération des fichiers csv -----")
    X_forecast_data_train = X_forecast(journées_train,numéros_stations,data_path)
    pd.DataFrame(X_forecast_data_train.reshape((len(X_forecast_data_train),-1))).to_csv(output_folder+'/2D_arome_train.csv')
    print("----- 2D_arome_train.csv : Done -----")
    X_station_data_train = X_station_train(data_path+"/X_station_train.csv", journées_train,coordonnees_stations,numéros_stations,altitude_stations)
    X_station_data_test = X_station_test(data_path+"/X_station_test.csv",coordonnees_stations,numéros_stations,altitude_stations)
    pd.DataFrame(X_station_data_train.reshape((len(X_station_data_train),-1))).to_csv(output_folder+'/X_station_data_train.csv')
    print("----- X_station_data_train.csv : Done -----")
    pd.DataFrame(X_station_data_test.reshape((len(X_station_data_test),-1))).to_csv(output_folder+'/X_station_data_test.csv')
    print("----- X_station_data_test.csv : Done -----")
    Y_data_train = Y_station(data_path+"/Y_train.csv", journées_train,numéros_stations)
    pd.DataFrame(Y_data_train.reshape((len(Y_data_train),-1))).to_csv(output_folder+'/Y_data_train.csv')
    print("----- Y_data_train.csv : Done -----")
    
    te = time.time()
    print("Temps écoulé environ", (te-ts)//60, "min" , (te-ts)%60,'s')
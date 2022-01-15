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
    
    #Génération des np.array de la baseline forecast et baseline observation
    print("----- Génération des np.array de la baseline forecast et baseline observation -----")
    baseline_forecast = []
    baseline_observation = []

    for row in csv.reader(open (data_path+"/Baseline_forecast_test.csv",'r')):
        baseline_forecast.append(row)
    for row in csv.reader(open (data_path+"/Baseline_observation_test.csv",'r')):
        baseline_observation.append(row)

    baseline_forecast = np.array(baseline_forecast)[1:]
    baseline_observation = np.array(baseline_observation)[1:]
    submission_format = baseline_observation[:,0]
    
    # Mise en forme des arrays
    baseline_f = np.concatenate((baseline_forecast, np.zeros((len(baseline_forecast),1))), axis=1)
    baseline_o = np.concatenate((baseline_observation, np.zeros((len(baseline_observation),1))), axis=1)

    for i in range(len(baseline_f)):
        baseline_f[i,2] = baseline_f[i,1].split('_')[1]
        baseline_f[i,1] = baseline_f[i,1].split('_')[0]
    for i in range(len(baseline_o)):
        baseline_o[i,2] = baseline_o[i,1]
        baseline_o[i,1] = baseline_o[i,0].split('_')[1]
        baseline_o[i,0] = baseline_o[i,0].split('_')[0]
    
    # Conversion des arrays du string au float
    baseline_f = np.reshape(np.array(list(map(float, baseline_f.flatten()))), baseline_f.shape)
    baseline_o = np.reshape(np.array(list(map(float, baseline_o.flatten()))), baseline_o.shape)
    
    # Création du array total regroupant la baseline forecast et la baseline observation pour tous les échantillons de test.
    baseline = np.ones((363,325,2))*-np.pi

    for i in range(len(baseline_f)):
        jour = int(baseline_f[i,2])
        station = np.where(numéros_stations == baseline_f[i,1])[0][0]
        baseline[jour, station, 0] = baseline_f[i,0]
    for i in range(len(baseline_o)):
        jour = int(baseline_o[i,1])
        station = np.where(numéros_stations == baseline_o[i,0])[0][0]
        baseline[jour, station, 1] = baseline_o[i,2]
    
    # Mise en commun des échantillons des baselines forecast et observation (échantillon, [baseline_f, baseline_o])
    baseline_test = np.copy(baseline)

    for i in range(baseline_test.shape[0]):
        for j in range(baseline_test.shape[1]):
            if np.count_nonzero(baseline_test[i,j] == -np.pi) != 0:
                baseline_test[i,j,0] = -2*np.pi

    baseline_test = baseline_test[baseline_test[:,:,0] != -2*np.pi]
    
    
    # Préparation des tenseurs
    print("----- Préparation des tenseurs : ce sera un peu long ! Soyez patient...-----")
    
    X_forecast_data_train = genfromtxt(data_path+'/2D_arome_train.csv', delimiter=',', skip_header = 1)
    X_forecast_data_train = np.delete(X_forecast_data_train, 0, axis=1).reshape((-1,325,24,7))
    print("----- X_forecast_data_train : Done ! Au suivant ... -----")
    
    X_forecast_data_test = genfromtxt(data_path+'/2D_arome_test.csv', delimiter=',', skip_header = 1)
    X_forecast_data_test = np.delete(X_forecast_data_test, 0, axis=1).reshape((-1,325,24,7))
    print("----- X_forecast_data_test : Done ! Au suivant ... -----")

    X_station_data_train = genfromtxt(data_path+'/X_station_data_train.csv', delimiter=',', skip_header = 1)
    X_station_data_train = np.delete(X_station_data_train, 0, axis=1).reshape((-1,325,24,10))
    print("----- X_station_data_train : Done ! Au suivant ... -----")
    
    X_station_data_test = genfromtxt(data_path+'/X_station_data_test.csv', delimiter=',', skip_header = 1)
    X_station_data_test = np.delete(X_station_data_test, 0, axis=1).reshape((-1,325,24,10))
    print("----- X_station_data_test : Done ! Au suivant ... -----")

    Y_data_train = genfromtxt(data_path+'/Y_data_train.csv', delimiter=',', skip_header = 1)
    Y_data_train = np.delete(Y_data_train, 0, axis=1).reshape((-1,325,1))
    print("----- Y_data_train : Done ! Enfin !-----")
    
    # supression des nan dans le np.array X_station_data_test
    col_mean = np.nanmean(X_station_data_test, axis = 0)*0-np.pi
    inds = np.where(np.isnan(X_station_data_test)) 
    X_station_data_test[inds] = np.take(col_mean, inds[1]) 
    
    X_forecast_data_train = np.concatenate((X_forecast_data_train, np.zeros((731,325,24,4))), axis=3)
    X_forecast_data_test = np.concatenate((X_forecast_data_test, np.zeros((363,325,24,4))), axis=3)

    for i in range(len(coordonnees_stations)):
        X_forecast_data_train[:,i,:,8] = coordonnees_stations[i,0]
        X_forecast_data_train[:,i,:,9] = coordonnees_stations[i,1]
        X_forecast_data_train[:,i,:,10] = altitude_stations[i]
        X_forecast_data_test[:,i,:,8] = coordonnees_stations[i,0]
        X_forecast_data_test[:,i,:,9] = coordonnees_stations[i,1]
        X_forecast_data_test[:,i,:,10] = altitude_stations[i]

    for i in range(len(journées_train)):
        X_forecast_data_train[i,:,:,7] = int(journées_train[i][5:7])
    for i in range(len(journées_test)):
        X_forecast_data_test[i,:,:,7] = id_month_test[i,1]
    
    # Matrice des identifiants de l'échantillon de test au format : (nb_jours_test, nb_stations, [jour_test, id_station, mois_test])
    id_data_test = np.zeros((363,325,3), dtype=float)

    for i in range(id_data_test.shape[0]):
        for j in range(id_data_test.shape[1]):
            id_data_test[i,j,0] = journées_test[i]
            id_data_test[i,j,1] = numéros_stations[j]
            id_data_test[i,j,2] = id_month_test[i,1]
        
    # X_station est de la forme : (mois, ff, t, td, hu, dd, precip, longitude, latitude, altitude)
    # On choisit de conserver les prédicteurs de station les plus présents dans l'ensemble des échantillons, soient les prédicteurs de t et precip.
    X_forecast_train = np.copy(X_forecast_data_train[1:])
    X_station_train = np.concatenate((np.reshape(np.copy(X_station_data_train[:-1,:,:,2]),(-1,325,24,1)), np.reshape(np.copy(X_station_data_train[:-1,:,:,6]),(-1,325,24,1))), axis=3)
    Y_train = np.copy(Y_data_train[1:])
    X_forecast_test = np.copy(X_forecast_data_test)
    X_station_test = np.concatenate((np.reshape(np.copy(X_station_data_test[:,:,:,2]),(-1,325,24,1)), np.reshape(np.copy(X_station_data_test[:,:,:,6]),(-1,325,24,1))), axis=3)
    
    
    # Création des tenseurs d'entrées d'entraînement du réseau de neurones.
    for i in range(Y_train.shape[0]):
        for j in range(Y_train.shape[1]):
            if np.count_nonzero(X_station_train[i,j] == -np.pi) + np.count_nonzero(Y_train[i,j] == -np.pi) + np.count_nonzero(X_forecast_train[i,j] == -np.pi)!= 0:
                X_forecast_train[i,j,0,0] = -2*np.pi
                X_station_train[i,j,0,0] = -2*np.pi
                Y_train[i,j,0] = -2*np.pi

    X_forecast_train = X_forecast_train[X_forecast_train[:,:,0,0] != -2*np.pi]
    X_station_train = X_station_train[X_station_train[:,:,0,0] != -2*np.pi]
    Y_train = Y_train[Y_train[:,:,0] != -2*np.pi]
    
    
    # Création des tenseurs d'entrées de test du réseau de neurones.
    for i in range(X_station_test.shape[0]):
          for j in range(X_station_test.shape[1]):
            if np.count_nonzero(X_station_test[i,j] == -np.pi) + np.count_nonzero(X_forecast_test[i,j] == -np.pi) != 0:
                X_forecast_test[i,j,0,0] = -2*np.pi
                X_station_test[i,j,0,0] = -2*np.pi
                id_data_test[i,j,0] = -2*np.pi
    X_forecast_test = X_forecast_test[X_forecast_test[:,:,0,0] != -2*np.pi]
    X_station_test = X_station_test[X_station_test[:,:,0,0] != -2*np.pi]
    
    # Conversion de la liste des identifiants (numéro_station__indice_jour_test) des échantillons de test du float au string.
    id_test_float = id_data_test[id_data_test[:,:,0] != -2*np.pi]
    id_test_str = []

    for i in range(len(id_test_float)):
        id_test_str.append(str(int(id_test_float[i,1]))+"_"+str(int(id_test_float[i,0])))

    id_test_str = np.asarray(id_test_str)
    
    # Normalisation des X_station
    mean_station = [np.mean(X_station_train[:,:,i]) for i in range(X_station_train.shape[-1])]
    std_station = [np.std(X_station_train[:,:,i]) for i in range(X_station_train.shape[-1])]
    for i in range(X_station_train.shape[-1]):
        X_station_train[:,:,i] -= mean_station[i]
        X_station_train[:,:,i] /= std_station[i]
        X_station_test[:,:,i] -= mean_station[i]
        X_station_test[:,:,i] /= std_station[i]
        
    # Normalisation des X_forecast
    mean_forecast = [np.mean(X_forecast_train[:,:,i]) for i in range(X_forecast_train.shape[-1])]
    std_forecast = [np.std(X_forecast_train[:,:,i]) for i in range(X_forecast_train.shape[-1])]
    for i in range(X_forecast_train.shape[-1]):
        X_forecast_train[:,:,i] -= mean_forecast[i]
        X_forecast_train[:,:,i] /= std_forecast[i]
        X_forecast_test[:,:,i] -= mean_forecast[i]
        X_forecast_test[:,:,i] /= std_forecast[i]
        
    
    # Entraînement du premier réseau : LSTM avec dropout sur X_station_train et X_forecast_train
    print("----- Entraînement du premier réseau : LSTM avec dropout sur X_station_train et X_forecast_train -----")
    model_1 = lstm_3_1()
    entrainement_NN(model_1, optimizer=keras.optimizers.Adam(learning_rate = 1e-4), loss=build_MAPE, 
                inputs=[X_station_train, X_forecast_train[:,:,:-4]], outputs=Y_train, 
                epochs=5, batch_size=170,folder = output_folder,name_model='LSTM_station_forecast')
    
    print("----- Weights sauvegardés dans OUTPUT : LSTM avec dropout sur X_station_train et X_forecast_train -----")
    
    # Entraînement du deuxième réseau : LSTM sur X_forecast_train uniquement
    print("----- Entraînement du deuxième réseau : LSTM sur X_forecast_train uniquement -----")
    model_2 = lstm_4()
    entrainement_NN(NN=model_2, 
                optimizer=keras.optimizers.Adam(learning_rate = 1e-4), loss=build_MAPE, 
                inputs=X_forecast_train[:,:,:-4], outputs=Y_train, 
                epochs=5, batch_size=170,folder = output_folder,name_model='LSTM_forecast')
    
    print("----- Weights sauvegardés dans OUTPUT : LSTM sur X_forecast_train uniquement -----")
    
    # Entraînement du troisième réseau : DENSE sur X_forecast_train uniquement
    print("----- Entraînement du troisième réseau : DENSE sur X_forecast_train uniquement -----")
    model_3 = sequential_3_5()
    entrainement_NN(NN=model_3, 
                optimizer=keras.optimizers.Adam(learning_rate = 1e-4), loss=build_MAPE, 
                inputs=X_forecast_train[:,:,:-4], outputs=Y_train, 
                epochs=5, batch_size=170,folder = output_folder,name_model='DENSE_forecast')
    print("----- Weights sauvegardés dans OUTPUT : DENSE sur X_forecast_train uniquement -----")
    
    # Entraînement du quatrième réseau : CONV1D sur X_forecast_train et prédicteurs constants
    print("----- Entraînement du quatrième réseau : CONV1D sur X_forecast_train et prédicteurs constants -----")
    model_4 = Convolution_5()
    entrainement_NN(NN=model_4, 
                optimizer=keras.optimizers.Adam(learning_rate = 1e-4), loss=build_MAPE, 
                inputs=X_forecast_train, outputs=Y_train, 
                epochs=5, batch_size=170,folder = output_folder,name_model='CONV1D_forecast_pred_const')
    print("----- Weights sauvegardés dans OUTPUT : CONV1D sur X_forecast_train et prédicteurs constants -----")
    
    # Prédiction des réseaux sur X_test, moyenne de ces 4 prédictions et sauvegarde du fichier final
    print("----- Prédiction des réseaux sur X_test et moyenne de ces 4 prédictions -----")
    Y_forecast_model_1_build_mape = model_1.predict([X_station_test, X_forecast_test[:,:,:-4]])
    Y_forecast_model_2_build_mape = model_2.predict(X_forecast_test[:,:,:-4])
    Y_forecast_model_3_build_mape = model_3.predict(X_forecast_test[:,:,:-4])
    Y_forecast_model_4_build_mape = model_4.predict(X_forecast_test)
    
    Y_forecast_moyenne_4_modeles = (Y_forecast_model_1_build_mape + Y_forecast_model_2_build_mape + Y_forecast_model_3_build_mape + Y_forecast_model_4_build_mape)/4
    
    print("----- Sauvegarde du fichier final à déposer sur Kaggle en cours...  -----")
    mise_en_forme_prediction_kagglev2(Y_forecast_moyenne_4_modeles,output_folder,submission_format,id_test_str)
    print("----- Done !  Retrouvez-le dans votre dossier OUTPUT et déposez-le sur Kaggle pour voir son score MAPE -----")
    
    te = time.time()
    print("Temps écoulé environ", (te-ts)//60, "min" , (te-ts)%60,'s')
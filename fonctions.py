############### Fonctions utilisées dans le traitement des données ################################
##########dans les procédures d'entraînement des réseaux et dans la génération du fichier final####


##################### Fonctions pour le traitement des données ##################### 
def X_forecast_jour(ds):
    # Conversion des prévisions modèle du format xarray.Dataset au format numpy.
    ds_np = np.delete(np.array((ds['u10'], ds['v10'], ds['t2m'], ds['d2m'], ds['r'], ds['tp'], ds['msl'])), 0, axis=1)
    
    # Redimensionnement du tenseur des prévisions modèle.
    prévisions_modèle = np.zeros((ds_np.shape[1], ds_np.shape[2], ds_np.shape[3], ds_np.shape[0]))
    for i in range(ds_np.shape[0]):
      prévisions_modèle[:,:,:,i] = ds_np[i,:,:,:]
    
    # Fonction qui renvoie à partir des coordonnées d'une station de mesure, les coordonnées des points de grille 
    # du modèle AROME qui encadrent la station, ainsi que l'indice dans la liste des prévisions modèle de ces points de grille.
    def chgt_base(point):
      x, y = point[0], point[1]

      z1 = (51.896 - x)/0.025
      ind_lat_inf, ind_lat_sup = int(z1), int(z1)+1
      lat_inf, lat_sup = np.round(51.896 - ind_lat_inf*0.025,3) , np.round(51.896 - ind_lat_sup*0.025, 3)

      z2 = (y + 5.842)/0.025
      ind_long_inf, ind_long_sup = int(z2), int(z2)+1
      long_inf, long_sup = np.round(-5.842 + ind_long_inf*0.025,3) , np.round(-5.842 + ind_long_sup*0.025, 3)

      return [[ind_lat_inf, lat_inf], [ind_lat_sup, lat_sup], [ind_long_inf, long_inf], [ind_long_sup, long_sup]]
    
    prévisions_stations = np.zeros((len(coordonnees_stations), prévisions_modèle.shape[0], prévisions_modèle.shape[-1]))
    
    # Création du tenseur des prévisions modèle interpolées sur les stations de mesure.
    for space in range(len(coordonnees_stations)):
      L = chgt_base(coordonnees_stations[space])

      coord_NO = [L[0][1],L[2][1]]
      coord_NE = [L[0][1],L[3][1]]
      coord_SO = [L[1][1],L[2][1]]
      coord_SE = [L[1][1],L[3][1]]

      tree = spatial.KDTree([(coordonnees_stations[space,0], coordonnees_stations[space,1])])
      dist_NO = tree.query(coord_NO)[0]
      dist_NE = tree.query(coord_NE)[0]
      dist_SO = tree.query(coord_SO)[0]
      dist_SE = tree.query(coord_SE)[0]

      s = 1/dist_NO**2 + 1/dist_NE**2 + 1/dist_SO**2 + 1/dist_SE**2

      for time in range(prévisions_modèle.shape[0]):
        for pred in range(prévisions_modèle.shape[-1]):
          prev_NO = prévisions_modèle[time, L[0][0], L[2][0], pred]
          prev_NE = prévisions_modèle[time, L[0][0], L[3][0], pred]
          prev_SO = prévisions_modèle[time, L[1][0], L[2][0], pred]
          prev_SE = prévisions_modèle[time, L[1][0], L[3][0], pred]

          prev = (prev_NO/dist_NO**2 + prev_NE/dist_NE**2 + prev_SO/dist_SO**2 + prev_SE/dist_SE**2) / s

          prévisions_stations[space, time, pred] = prev
    
    return prévisions_stations


# fonction qui traite les fichiers X_forecast.nc du modèle 2D_arome et qui les convertit en un np.array de la forme :
# (nb_jours, nb_stations, nb_heures, nb_prédicteurs)

def X_forecast(journées):
    X_forecast = np.zeros(((1,len(numéros_stations),24,7)))
    for i in range(len(journées)):
        ############# train
        journee = journées[i]
        fichier = journee[:4]+journee[5:7]+journee[8:10]
        print(journee)
        ################
        ############# test
        #fichier = str(journées[i])
        #print("journée test :",i)
        ################
        if os.path.exists(path+"/2D_arome/2D_arome_"+fichier+".nc") == False:
            print(1)
            X_forecast = np.concatenate((X_forecast, np.ones((1,len(numéros_stations),24,7))*(-np.pi)), axis=0)
        else:
            ds = xr.load_dataset(path+"/2D_arome/2D_arome_"+fichier+".nc", engine="netcdf4")
            if len(ds.data_vars) != 9:
                print(1.5)
                X_forecast = np.concatenate((X_forecast, np.ones((1,len(numéros_stations),24,7))*(-np.pi)), axis=0)
            else:
                pred = len(ds['u10']) + len(ds['v10']) + len(ds['t2m']) + len(ds['d2m']) + len(ds['r']) +  len(ds['tp']) + len(ds['msl'])
                latitude = len(ds.latitude)
                longitude = len(ds.longitude)
                #valid_time = len(ds.Id)   ##### test
                valid_time = len(ds.valid_time)   #### train
                valid_shape = [pred/25, latitude, longitude, valid_time]

                if valid_shape != [7,227,315,25]:
                    print(2)
                    X_forecast = np.concatenate((X_forecast, np.ones((1,len(numéros_stations),24,7))*(-np.pi)), axis=0)
                else:
                    X_forecast_day = X_forecast_jour(ds).reshape((1,len(numéros_stations),24,7))

                    if np.isnan(X_forecast_day).sum() != 0:
                        print(3)
                        X_forecast = np.concatenate((X_forecast, np.ones((1,len(numéros_stations),24,7))*(-np.pi)), axis=0)
                    else:
                        print(4)
                        X_forecast = np.concatenate((X_forecast, X_forecast_day), axis=0)
                        
    X_forecast = np.delete(X_forecast, 0, axis=0)
    return X_forecast



def X_station_train(fichier_csv, journées):
    
    # Importation des observations des stations de mesure sous forme de fichier .txt,
    # et production d'un np.array X_station des données brutes extraites.
    # Les données extraites sont de la forme : (number_station, date, heure, ff, t, td, hu, dd, precip, Id).
    X_station = []
    for row in csv.reader(open(fichier_csv)):
      X_station.append(row)
    X_station = np.asarray(X_station)
    
    # Redimensionnement du **np.array** *X_station* des observations extraites.
    # Le nouveau tenseur sera de la forme : 
    # (number_station, mois, jour, heure, ff, t, td, hu, dd, precip, longitude, latitude, altitude)
    X_station = np.delete(X_station, 0, axis=0)
    X_station = np.delete(X_station, -1, axis=1)
    X_station = np.insert(X_station, 1, np.zeros((X_station.shape[0])), axis=1)
    X_station = np.insert(X_station, 1, np.zeros((X_station.shape[0])), axis=1)
    X_station = np.insert(X_station, X_station.shape[-1], np.zeros((X_station.shape[0])), axis=1)
    X_station = np.insert(X_station, X_station.shape[-1], np.zeros((X_station.shape[0])), axis=1)
    X_station = np.insert(X_station, X_station.shape[-1], np.zeros((X_station.shape[0])), axis=1)
    
    # Mise en forme des données du tenseur X_station.
    for i in range(len(X_station)):
      X_station[i,1] = X_station[i,3].split(' ')[0]
      X_station[i,3] = X_station[i,3].split(' ')[1]
    for i in range(len(X_station)):
      X_station[i,2] = np.where(journées == X_station[i,1])[0][0]
      X_station[i,1] = np.float(X_station[i,1].replace('-',''))//100%100
      X_station[i,3] = np.float(X_station[i,3].replace(':',''))//10000
    
    # On complète le tenseur, en mettant à la place des données manquantes sous la forme de '',
    # la valeur -pi sous forme de string.
    X_station = list(X_station)
    for i in range(len(X_station)):
      for j in range(len(X_station[0])):
        if X_station[i][j] == '':
          X_station[i][j] = X_station[i][j]+str(-np.pi)
    X_station = np.array(X_station)
    
    # On convertit enfin tous les éléments du tenseur X_station pour le moment constitué, du string au float.
    X_station = np.reshape(np.array(list(map(float, X_station.flatten()))), X_station.shape)
    
    # Pour notre problématique de prévision des précipitations pour le jour suivant,
    # nous choisissons de considérer que les observations obtenues à 00:00:00 une journée J,
    # au lieu d'être stockées à la date de ce jour J, seront en réalité stockées à la date du jour J-1.
    # On diminue donc d'une unité l'indice du jour correspondant aux observations effectuées à minuit.
    for i in range(len(X_station)):
      if X_station[i,3] == 0:
        X_station[i,2] -= 1
    
    # On complète finalement les trois dernières colonnes précédemment créées et restées vides
    # par les prédicteurs relatifs à la géographie de la station de mesure : la longitude, la latitude, et l'alitude.
    for i in range(len(X_station)):
      station = np.where(data[:,0] == X_station[i,0])[0][0]
      X_station[i,10] = coordonnees_stations[station,0]
      X_station[i,11] = coordonnees_stations[station,1]
      X_station[i,12] = altitude_stations[station]
        
    # Production d'un np.array X_station_data qui stockera toutes les données contenues dans le tenseur X_station
    # de manière ordonné selon l'indice du jour, l'indice de la station de mesure, l'heure et les prédicteurs considérés,
    # de la même façon qu'est agencé le tenseur numpy X_forecast_data.
    X_station_data = np.ones((len(journées), len(numéros_stations), 24, 10))*(-np.pi)
    for i in range(X_station.shape[0]):
      jour = int(X_station[i,2])
      heure = int(X_station[i,3])
      station = np.where(numéros_stations == X_station[i,0])[0][0]
      for pred in range(10):
        if pred == 0:
          X_station_data[jour, station, heure, pred] = X_station[i,1]
        else:
          X_station_data[jour, station, heure, pred] = X_station[i, pred+3]
    
    return X_station_data



def X_station_test(fichier_csv):
  X_station = []
  for row in csv.reader(open(fichier_csv)):
    X_station.append(row)
  X_station = np.asarray(X_station)

  X_station = np.concatenate((X_station, np.zeros((len(X_station),5))), axis=1)

  X_station_norm = np.copy(X_station)
  for i in range(len(X_station)):
    print(i)
    X_station_norm[i,1] = X_station[i,6]
    if len(X_station[i,7].split('_'))==3:
      X_station_norm[i,0] = X_station[i,7].split('_')[0]
      X_station_norm[i,2] = X_station[i,7].split('_')[1]
      X_station_norm[i,3] = X_station[i,7].split('_')[2]
    else:
      X_station_norm[i,0] = ''
      X_station_norm[i,2] = ''
      X_station_norm[i,3] = ''
    X_station_norm[i,4] = X_station[i,4]
    X_station_norm[i,5] = X_station[i,3]
    X_station_norm[i,6] = X_station[i,2]
    X_station_norm[i,7] = X_station[i,1]
    X_station_norm[i,8] = X_station[i,0]
    X_station_norm[i,9] = X_station[i,5]

  for i in range(len(X_station_norm)):
    if X_station_norm[i,3] == 0:
      X_station_norm[i,2] -= 1

  for i in range(len(X_station_norm)):
    station = np.where(data[:,0] == X_station_norm[i,0])[0][0]
    X_station_norm[i,10] = coordonnees_stations[station,0]
    X_station_norm[i,11] = coordonnees_stations[station,1]
    X_station_norm[i,12] = altitude_stations[station]
  
  X_station_test = np.ones((363, len(numéros_stations), 24, 10))*(-np.pi)
  for i in range(X_station_norm.shape[0]):
    jour = int(X_station_norm[i,2])
    heure = int(X_station_norm[i,3])
    station = np.where(numéros_stations == X_station_norm[i,0])[0][0]
    for pred in range(10):
      if pred == 0:
        X_station_test[jour, station, heure, pred] = X_station_norm[i,1]
      else:
        X_station_test[jour, station, heure, pred] = X_station_norm[i, pred+3]

  return X_station_test


def Y_station(fichier_csv, journées):
    
    Y_station = []
    for row in csv.reader(open(fichier_csv)):
      Y_station.append(row)
    Y_station = np.asarray(Y_station)
    
    Y_station = Y_station[:,:3]
    Y_station = Y_station[1:,:]
    
    for i in range(len(Y_station)):
      Y_station[i,0] = np.where(journées == Y_station[i,0])[0][0]
      Y_station[i,1] = np.where(numéros_stations == float(Y_station[i,1]))[0][0]
        
    Y_station = list(Y_station)
    for i in range(len(Y_station)):
      if Y_station[i][2] == '':
        Y_station[i][2] += str(-np.pi)
    Y_station = np.array(Y_station)
    
    Y_station = np.reshape(np.array(list(map(float, Y_station.flatten()))), Y_station.shape)
    
    Y_station_data = np.ones((len(journées), len(numéros_stations), 1))*(-np.pi)
    for i in range(Y_station.shape[0]):
      jour = int(Y_station[i,0])
      station = int(Y_station[i,1])
      Y_station_data[jour,station] = Y_station[i,2]
    
    return Y_station_data



##################### Fonctions pour les réseaux de neurones ##################### 

# Création de la Loss function pour tous les réseaux
def build_MAPE(y_true, y_pred):
  loss = tf.reduce_mean(tf.abs((y_true - y_pred)/(y_true + 1)), axis=-1) * 100
  return loss



def entrainement_NN(NN, optimizer, loss, inputs, outputs, epochs, batch_size,verbose = 1,data_path):
  
  NN.compile(optimizer=optimizer,
              loss=loss,
              metrics=build_MAPE)
  
  history = NN.fit(inputs,outputs,
                   epochs=epochs,
                   batch_size=batch_size,
                   verbose = verbose,
                   validation_split=0.2)
  

    

def lstm_3_1():  
  encoder_inputs = Input(shape=(24, 2), name='encoder_inputs')

  encoder_lstm = LSTM(32, dropout=0.2,recurrent_dropout=0.2, return_sequences=False, name='lstm_1')(encoder_inputs)
  encoder_lstm_2 = Dense(7, activation='relu', name='dense_1')(encoder_lstm)
  encoder_lstm_3 = Reshape((1,7))(encoder_lstm_2)

  decoder_inputs = Input(shape=(24, 7), name='decoder_inputs')

  decoder_concat = Concatenate(axis=1)([encoder_lstm_3, decoder_inputs])

  decoder_lstm = LSTM(128,dropout=0.2, recurrent_dropout=0.2, return_sequences=True, name='lstm_2')(decoder_concat)
  decoder_lstm_2 = Lambda(lambda x: x[:, 1:, :])(decoder_lstm)

  decoder_lstm_3 = TimeDistributed(Dense(32, activation='relu'))(decoder_lstm_2)
  decoder_lstm_4 = TimeDistributed(Dense(1, activation='relu'))(decoder_lstm_3)
  decoder_lstm_4 = Flatten()(decoder_lstm_4)

  decoder_outputs = sum(decoder_lstm_4, axis=1, keepdims=True)

  return Model([encoder_inputs, decoder_inputs], decoder_outputs, name='LSTM_3_1')



def sequential_3_5():
  inputs = Input(shape=(24,7), name='inputs')
  x = Flatten()(inputs)
  x = Dense(64, activation='relu', name='dense_1')(x)
  x = Dense(128, activation='relu', name='dense_2')(x)
  x = Dropout(0.2)(x)
  x = Dense(128, activation='relu', name='dense_3')(x)
  x = Dropout(0.2)(x)
  x = Dense(64, activation='relu', name='dense_4')(x)
  x = Dense(32, activation='relu', name='dense_5')(x)
  outputs = Dense(1, activation='relu', name='dense_6')(x)
  return Model(inputs, outputs, name='Sequential_3_5')



def Convolution_5():
  inputs = Input(shape=(24,11), name='inputs')
  x = Conv1D(32,7, activation='relu')(inputs)
  x = MaxPooling1D(2)(x)
  x = Conv1D(32,7, activation='relu')(x)
  x = tf.keras.layers.GlobalMaxPooling1D()(x)
  outputs = Dense(1, activation='relu', name='dense_6')(x)
  return Model(inputs, outputs, name='Convolution_5')




##################### Fonctions pour le fichier final ##################### 


# Fonction Mise en forme des prévisions de précipitations.
def mise_en_forme_prediction_kagglev2(Y_forecast_model,data_path):
  predictions = np.empty((85140,1))
  predictions[:] = np.nan

  for i in range(len(predictions)):
    id_sf = baseline_observation[i,0]
    ind = np.where(id_test_str == id_sf)[0]
    if ind.shape[0] == 1:
      n = ind[0]
      prediction = Y_forecast_model[n]
      predictions[i] = prediction
  predictions += 1  # on rajoute le 1 comme demandé sur kaggle

  # remplacement des NaN dans le np.array predictions des zéros.
  col_mean = np.nanmean(predictions, axis = 0)*0 +1
  inds = np.where(np.isnan(predictions))
  predictions[inds] = np.take(col_mean, inds[1])
  
  # Création du fichier final de soumission des résultats du défi IA sur Kaggle.
  submission_github = np.concatenate((np.reshape(baseline_observation[:,0], (-1,1)), predictions), axis=1)

  pd.DataFrame(submission_github.reshape((len(submission_github),-1))).to_csv(data_path+'/predictions_ENM_Les_Rainettes.csv')

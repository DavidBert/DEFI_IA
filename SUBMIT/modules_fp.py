import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime
import joblib

def check_one_station(df_one, first_date_ordinal, last_date_ordinal):
    '''
    This function is applied to each station and check two conditions 
    to be included in training to be included in training:
    First; if all data is available in this station and if there is
    less than 25 percent of Nans values in precip
    '''      
    dataset = []

    days = np.array([d.day for d in df_one["date"]])
    months = np.array([d.month for d in df_one["date"]])
    years = np.array([d.year for d in df_one["date"]])

    condition_select_1=True  # check that all days are available
    condition_select_2 = False # check less than a threshold of Nans 
    
    # we dont consider last date because its not used for training, only for labeling
    for i in range(first_date_ordinal,last_date_ordinal-1):
        date1 = datetime.date.fromordinal(i)
        day_mask = (days==date1.day) & (months==date1.month) & (years==date1.year)
        if np.count_nonzero(day_mask)==0:
            condition_select_1=False
            break
        df_one_day = df_one[day_mask]
        np_array = df_one_day[['ff','t','td','hu','dd','precip']].to_numpy().ravel() 
        dataset.append(np_array)
    
    if condition_select_1:
        dataset = np.array(dataset)
    
    if condition_select_1:
        nan_p100 = 100*np.count_nonzero(np.isnan(dataset[:,5]))/dataset.shape[0]
        condition_select_2 = nan_p100<20 # less than 25 percent of Nan in rain values

    select_station=False

    if (condition_select_1 & condition_select_2):
        select_station = True
    return select_station
    

def select_stations(first_date, last_date, fname):
    '''
    Returns a list with accepted stations to train 
    (with all data available and less tha 25 percent of nans in precipitation)
    '''
    df = pd.read_csv(fname,parse_dates=['date'],infer_datetime_format=True)

    first_date_ordinal = first_date.toordinal()
    last_date_ordinal = last_date.toordinal()

    selected_stations = []

    # ALL STATIONS
    all_stations = df["number_sta"].unique()

    # ONE STATION TREATMENT
    for idx_sta in range(len(all_stations)):
    #for idx_sta in range(20):
    # open one station 
        number_sta = all_stations[idx_sta]
        df_one = df[df["number_sta"]==number_sta] # df containg one station

        # check if this station has all days and less than 25 percent of Nans
        select_station = check_one_station(df_one,first_date_ordinal,last_date_ordinal)
        
        if select_station:
            selected_stations.append(number_sta)

    return selected_stations


def create_X_one_station(first_date, last_date, df_one):
    
    first_date_ordinal = first_date.toordinal()
    last_date_ordinal = last_date.toordinal()
        
    days = np.array([d.day for d in df_one["date"]])
    months = np.array([d.month for d in df_one["date"]])
    years = np.array([d.year for d in df_one["date"]])

    X_one_station = []

    # we dont consider last day because its not used for training, only for labeling
    for i in range(first_date_ordinal,last_date_ordinal-1):
        date1 = datetime.date.fromordinal(i)
        day_mask = (days==date1.day) & (months==date1.month) & (years==date1.year)
        df_one_day = df_one[day_mask]
        np_array = one_day_handling(df_one_day, date1.month) 
        X_one_station.append(np_array)

    return X_one_station


def create_y_one_station(first_date, last_date, df_one, test=False):

    df_precip = df_one[{"number_sta","date","precip"}]
    df_precip.set_index('date',inplace = True)  

    #compute the accumulated rainfall as 24*(daily mean) to compensate nan values
    df_precip = df_precip.groupby('number_sta').resample('D').agg(pd.Series.mean)
    df_precip['precip'] *= 24.

    if test==False: # so is train or validation
      # we dont take the first day as is only used for training but not as label
      mask_dates = (df_precip.index.levels[1]>first_date) &  (df_precip.index.levels[1]<last_date) 
      df_precip = df_precip.iloc[mask_dates]
    
    y = df_precip["precip"].tolist()

    return y


def create_and_save_dataset(fname, selected_sta, path_results, first_date, last_date):
    
    df_train = pd.read_csv(fname,parse_dates=['date'],infer_datetime_format=True)
    # Add some features combination of columns
    df_train = add_features_df(df_train)
    X_all_stations = []
    y_all_stations = list([])
    for number_sta in selected_sta:
        
        # dataframe with one station information
        print('Pretreatement, treating Station:' + str(number_sta))
        df_one = df_train[df_train["number_sta"]==number_sta]
        
        # generate our processed database for one station
        X_one_station = create_X_one_station(first_date, last_date, df_one)
        y_one_station = create_y_one_station(first_date, last_date, df_one)

        X_all_stations += X_one_station
        y_all_stations = y_all_stations + y_one_station


    # data scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    X_scaled = sc.fit_transform(X_all_stations)
    scaler_filename = path_results + "/scaler.save"
    joblib.dump(sc, scaler_filename)

    # Nan replacement by each feature mean
    mean_features_values = np.nanmean(X_scaled, axis=0)
    idx_nan = np.where(np.isnan(X_scaled))
    X_scaled[idx_nan] = np.take(mean_features_values, idx_nan[1])

    # Y nan replacement by mean value and add offset of 1 (because scoring rule)
    y_all_stations = np.array(y_all_stations)
    mean_y_values = np.nanmean(y_all_stations)
    idx_nan = np.where(np.isnan(y_all_stations))
    y_all_stations[idx_nan] = mean_y_values
    y_all_stations = [y_i+1. for y_i in y_all_stations]

    # Split in TRAIN and VALIDATION sets 85 and 15 percent respectively
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_all_stations,
                                                      test_size=0.20, random_state=42)

    # save dataset
    np.save(path_results + '/X_train.npy', X_train)
    np.save(path_results + '/y_train.npy', y_train)
    np.save(path_results + '/X_val.npy', X_val)
    np.save(path_results + '/y_val.npy', y_val)

def one_day_handling(df_one_day, month, test=False):
    features_hour = 13 # parameters per hourly data
    daily_rain = df_one_day['precip'].sum()
    if test==False:
      one_day_array = df_one_day[['ff','t','td','hu','dd','precip','hu2','hu_T',
                                  'hu_T_Td','t_minus_td','t_div_td','humidex',
                                  'windchill']].to_numpy().ravel()
    
    if test==True:
      one_day_array=[]
      for hour in range(24):
        #try:
        one_hour = df_one_day[df_one_day['hour']==str(hour)][['ff','t','td',
                                                              'hu','dd','precip',
                                                              'hu2','hu_T','hu_T_Td',
                                                              't_minus_td','t_div_td',
                                                              'humidex','windchill']].to_numpy()
        if one_hour.shape == (1,features_hour):
          one_day_array.append(one_hour) 
        else:
          one_day_array.append(np.nan*np.zeros((1,features_hour)))     
      one_day_array = np.concatenate(one_day_array, axis=None)
    
    # add month and cumulated day rain
    to_insert_daily = np.array([month, daily_rain])
    one_day_array = np.insert(one_day_array, 0,to_insert_daily)
      
    return one_day_array

def load_dataset(path_results, fX_train = '/X_train.npy', fy_train = '/y_train.npy', 
                 fX_val='/X_val.npy',fy_val='/y_val.npy',):
    X_train = np.load(path_results + fX_train)
    y_train = np.load(path_results + fy_train)
    X_val = np.load(path_results + fX_val)
    y_val = np.load(path_results + fy_val)
    if X_train.shape[0] != len(y_train):
        raise Exception("ERROR : X_data and y_data dont have the same length ! ")
    return X_train, y_train, X_val, y_val


def create_model(features_size,loss_f='mean_absolute_percentage_error', 
                 units=50, mod_='regr',input_size=1):
    if mod_ == 'lstm':  
      # LSTM IMPLEMENTED BUT NOT STUDIED YET    
      model = Sequential()
      model.add(LSTM(units=units, return_sequences=True, 
                     input_shape=(input_size, features_size)))
      model.add(Dropout(0.2))
      model.add(LSTM(units=units, return_sequences=True))
      model.add(Dropout(0.2))
      model.add(LSTM(units=units, return_sequences=True))
      model.add(Dropout(0.2))
      model.add(LSTM(units=units))
      model.add(Dropout(0.2))
      model.add(Dense(units=1))
      model.compile(optimizer='adam', loss='mean_squared_error')
    elif mod_== 'regr':
      model = Sequential()
      model.add(Dense(300, input_dim=features_size, activation= "relu"))
      model.add(Dense(300, activation= "relu"))
      model.add(Dense(100, activation= "relu"))
      model.add(Dense(1))
      model.compile(optimizer='adam', loss='mean_absolute_percentage_error')
    
    return model
    
# MAPE Score Calculator
def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    if len(actual)<5000: # to avoid runout memory
      return np.mean(np.abs((actual - pred) / actual)) * 100
    else:
      return(np.mean(np.abs((actual[:5000] - pred[:5000]) / actual[:5000])) * 100)

# Create and save test set (TODO: improve implementetion and efficacity)
def create_and_save_test_set(path_data, path_results, scaler_filename):
    fname_test = path_data + '/Test/Test/X_station_test.csv'
    df_test = pd.read_csv(fname_test)

    # Add some features combination of columns
    df_test = add_features_df(df_test)
    
    # Check month of each day to add as a feature
    f_id_month = path_data +  '/Test/Test/Id_month_test.csv'
    df_id_month = pd.read_csv(f_id_month)

    # add station ID, day and hour to facilitate manipulation
    df_tmp = df_test['Id'].str.split('_',expand=True)
    df_test['number_sta']=df_tmp[0]
    df_test['day']=df_tmp[1]
    df_test['hour']=df_tmp[2]

    # ALL STATIONS
    all_stations = df_test["number_sta"].unique()
    X_all_stations = []
    index_id = []  # valid index to compare with baseline predictions
    for number_sta in all_stations:
        print('num_sta treating in test creation:',number_sta)
        df_one = df_test[df_test["number_sta"]==number_sta]
        X_station = []
        all_days = df_one["day"].unique()
 
        for dd in all_days:
            index_id.append(str(number_sta)+'_'+str(dd))
            df_one_day = df_one[df_one["day"]==dd]
            month = df_id_month[df_id_month["day_index"]==int(dd)].month.item()
            array_one_day = one_day_handling(df_one_day, month, test=True)
            X_station.append(array_one_day)
            
            if len(array_one_day) != 314: 
                print('Array one day',array_one_day.shape, array_one_day)
                raise Exception("ERROR : Features vector not length 314 ! ")
        X_all_stations += X_station

        # data scaling with scaler fitted during training with train dataset
        sc = joblib.load(scaler_filename) 
        X_scaled = sc.transform(X_all_stations)
        
        # Nan replacement by each feature mean
        mean_features_values = np.nanmean(X_scaled, axis=0)
        idx_nan = np.where(np.isnan(X_scaled))
        X_scaled[idx_nan] = np.take(mean_features_values, idx_nan[1])  
    
    np.save(path_results + 'X_test.npy', X_scaled)
    
    return X_scaled, np.array(index_id)

def idx2predict(index_id, fobs):
    # return valid index found in available baseline_obs
    df_baseline = pd.read_csv(fobs)
    index_baseline = df_baseline["Id"].unique()
    
    _, i_to_predict, _ = np.intersect1d(index_id, index_baseline,
                                        assume_unique=True, return_indices=True)
    return i_to_predict

def write_log(file_name,txt ):
    # Open a file with access mode 'a'
    file_object = open(file_name, 'a')
    # Append 'hello' at the end of file
    file_object.write(str(txt))
    # Close the file
    file_object.close()

def add_features_df(df):
    # add different expertise features 
    df['hu2']=df['hu']**2
    df['hu_T']=df['hu']*df['t']
    df['hu_T_Td']=df['hu_T']*df['td']
    df['t_minus_td']=df['t']-df['td']
    df['t_div_td']=df['t']/df['td']
    df['humidex']=df['t']+.5555*(6.11*np.exp(5417.753*(1./273.15-1/(273.15+df['td'])))-10.)
    df['windchill']=13.12 + 0.6215*df['t'] - 11.37*df['ff']**.16 + 0.3965*df['t']*df['ff']**.16 
    return df
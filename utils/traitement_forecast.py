import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt


    
def distance(lat1, long1, lat2, long2):
    return np.sqrt((lat1-lat2)**2 + (long1-long2)**2)


def forecast_plus_proche(lat_s, long_s, data, K=1):
    '''
    inputs: 
    lat_s : latitude de la station
    long_s : longitude de la station
    data: xarray contenant les données de prévision 
    K: le nombre de données que l'on considère, au début fixé à 1

    outputs:
    ind: les indices des K points les plus proches
    dist_min: vecteur de taille K contenant la distance entre la station et les K points les plus proches
    '''


    lat = data["latitude"].values
    long = data["longitude"].values

    # construction de la matrice des distances à la station
    dist = np.zeros((len(lat), len(long)))

    for i in range(len(lat)):
        for j in range(len(long)):
            dist[i,j]= distance(lat_s, long_s, lat[i], long[j])

    # indices de la prévision la plus proche de la station
    ind = np.zeros((K,2),dtype="int")
    dist_min = np.zeros(K)
    for k in range(K):
        ind[k,:]= np.argwhere(dist == np.min(dist))[0]
        dist_min[k] = np.min(dist)
        dist[int(ind[k,0]), int(ind[k,1])] = 100000


    return ind, dist_min


def forecast_plus_proche_global(coords, data, K):
    '''
    inputs: 
    coords : dataframe des coordonnées spatiales des stations en fonction de leur numéro
    data : xarray contenant les prévisions 
    K: nombre de points voulus pour chaque station

    outputs :
    df: dataframe contenant le numéro des stations, les indices des K points les plus proches, les distances entre la station
    et les K points les plus proches.
    '''


    df = pd.DataFrame({"number_sta": coords["number_sta"]})
    ind = []
    dist_min = []

    for n in df['number_sta']:

        #print("station : ", n)

        lat_s = coords[coords["number_sta" ]== n]["lat"].values[0]
        lon_s = coords[coords["number_sta" ]== n]["lon"].values[0]
        A,B = forecast_plus_proche(lat_s = lat_s, long_s=  lon_s, data = data, K=K)
        ind.append(A)
        dist_min.append(B)


    df["ind"] = ind
    df["dist_min"] = dist_min
    return df

def prevision(ind, dist_min, data, p, var):
    '''
    inputs:
    ind : indices des points les plus proches
    dist_min : distance avec les points les plus proches
    p: paramètre d'interpolation
    data: xarray contenant les prévisions
    var: tableau de string, variable à prédire

    outputs:
    precip : vecteur des prévisions de var pour une station par interpolation spatiale
    '''
    
    precip = np.zeros(len(var))
    
    for i in range(0,len(var)) : 

        # calcul de la prévision pour 1 station par interpolation spatiale

        x = pd.to_datetime(data.time.values)
        if  x.strftime("%Y%m%d") == dt.datetime(2016,1,1).strftime("%Y%m%d"):
            precip[i] = None 

        else :
            try:
                if var[i] == "tp" : #si la variable est "tp" on prend la valeur à minuit 
                    aux = data[var[i]].values[-1, ind[:,0], ind[:,1]]
                else : #sinon on fait la moyenne sur les valeurs
                    aux = data[var[i]].values[:, ind[:,0], ind[:,1]].mean(axis=0)
                    
                K = len(dist_min)
                precip[i] = sum(1/(dist_min[k]**p)*aux[k] for k in range(K)) / sum(1/(dist_min[k]**p) for k in range(K))

            except: 
                precip[i] = None
           

    return precip


def chargement_data_train(date, path, model) : 
    '''
    inputs:
    date : type datetime, date du xarray à charger
    path: chemin d'accès
    model: type string, modele choisit'''

    fname = path + "%s_%s.nc" %(model, date.strftime("%Y%m%d"))
    data = xr.open_dataset(fname)

    return data

def chargement_data_test(date, path, model) : 
    '''
    inputs:
    date : type datetime, date du xarray à charger
    path: chemin d'accès
    model: type string, modele choisit'''

    fname = path + "%s_%s.nc" %(model, str(date))
    #print(fname)
    data = xr.open_dataset(fname)
    return data


def add_prevision(p,df_distance, df_X, path, model,var, bool_train = True):
    '''
    inputs:
    df_distance: dataframe contenant number_sta, indices des K points les plus proches, distance entre les K points et number_sta
    df_X: dataframe contenant a minima number_sta et date, ordonné par date puis number_sta
    p: type float, paramètre d'interpolation
    path: type string, chemin d'accès
    model: type string, 'arome' / 'arpege'
    bool_train: booleen true = train, false = test
    var: variable à ajouter string

    outputs: 
    df_X: type dataframe avec colonne 'forecast' ajoutée à df_train
    '''
    prev = []

    for d in df_X['date'].unique():  
        
        try :
            #print('date :', d)
            if bool_train : 
                data = chargement_data_train(pd.to_datetime(d) + dt.timedelta(days = 1), path, model)
            else : 
                data = chargement_data_test(d, path, model )
                
        except : #exception si il n'y a pas de fichier forecast à cette date
            #print("pas de fichier forecast à la date :", d)
            for nb in df_X[df_X['date']==d]['number_sta'] : 
                prev.append(None)
        else: 
            for nb in df_X[df_X['date']==d]['number_sta'] : 
                ind = df_distance[df_distance["number_sta"]== nb]["ind"].values[0]
                dist_min = df_distance[df_distance["number_sta"]== nb]["dist_min"].values[0]
                precip = prevision(ind, dist_min, data, p,var)
                prev.append(precip)
    

    A = np.zeros((df_X.shape[0], len(var)))
    for i in range(len(prev)): 
        A[i, :] = prev[i]

    for v in range(len(var)):
        df_X["forecast_"+model+"_"+var[v]] = A[:,v]

    return df_X


def calcul_distance(coords, path, model, K):
    '''
    inputs:
    coords: type dataframe, contient longitude, latitude et nb_sta des stations
    path: chemin d'accès des données météonet (.nc)
    model: modele de prévision choisit 'arome' ou 'arpege'
    K: nombre de points les plus proches de la station choisit
   
    outputs:
    df_distance :  dataframe contenant number_sta + indice des K points les plus proches + distance K points à la station
    '''

    #chargement d'un xarray d'une date arbitraire pour pouvoir créer df_distance
    file_date= dt.datetime(2016,2,1)
    fname = path + "%s_%s.nc" %(model, file_date.strftime("%Y%m%d"))
    data = xr.open_dataset(fname)

    df_distance = forecast_plus_proche_global(coords, data, K=K)


    return df_distance
    
    
def prevision_3D(ind, dist_min, data, p) : 
    precip = np.zeros(len(data.heightAboveGround))

    for i in range(0,len(data.heightAboveGround)) : 

        # calcul de la prévision pour 1 station par interpolation spatiale

        x = pd.to_datetime(data.time.values)
        if  x.strftime("%Y%m%d") == dt.datetime(2016,1,1).strftime("%Y%m%d"):
            precip[i] = None 

        else :
            try:

                aux = data.pres.values[-1,i, ind[:,0], ind[:,1]]
                #print(aux)

                K = len(dist_min)
                precip[i] = sum(dist_min[k]**p*aux[k] for k in range(K)) / sum(dist_min[k]**p for k in range(K))


            except: 
                precip[i] = None
                #print("pas la variable ", var, " dans les données à la date :", x)

    return precip


def add_prevision_3D(p,df_distance, df_X, path, model, bool_train = True):
    '''
    inputs:
    df_distance: dataframe contenant number_sta, indices des K points les plus proches, distance entre les K points et number_sta
    df_X: dataframe contenant a minima number_sta et date, ordonné par date puis number_sta
    p: type float, paramètre d'interpolation
    path: type string, chemin d'accès
    model: type string, 'arome' / 'arpege'
    bool_train: booleen true = train, false = test
    
    outputs: 
    df_X: type dataframe avec colonne 'forecast' ajoutée à df_train
    '''
    prev = []

    for d in df_X['date'].unique():  
        
        try :
            #print('date :', d)
            if bool_train : 
                data = chargement_data_train(pd.to_datetime(d)+ dt.timedelta(days = 1), path, model)
            else : 
                data = chargement_data_test(d+1, path, model )

        except : #exception si il n'y a pas de fichier forecast à cette date
            #print("pas de fichier forecast à la date :", d)
            for nb in df_X[df_X['date']==d]['number_sta'] : 
                prev.append(None)
            print("erreur")
        else: 
            for nb in df_X[df_X['date']==d]['number_sta'] : 
                ind = df_distance[df_distance["number_sta"]== nb]["ind"].values[0]
                dist_min = df_distance[df_distance["number_sta"]== nb]["dist_min"].values[0]
                precip = prevision_3D(ind, dist_min, data, p)
                prev.append(precip)
    

    if bool_train : 
        fname = path + "%s_%s.nc" %(model, dt.datetime(2016,4,1).strftime("%Y%m%d"))
    else : 
        fname = path + "%s_%s.nc" %(model, '0')
        
    data = xr.open_dataset(fname)
    
    h = data.heightAboveGround.values
    A = np.zeros((df_X.shape[0], len(h)))
    for i in range(len(prev)): 
        A[i, :] = prev[i]

        

    for i in range(len(h)):
        df_X["forecast_"+model+"_"+str(h[i])] = A[:,i]

    return df_X

    
def fill_nan_test(path, model , p, df_distance, df_X):
    '''
    inputs:
    path: chemin d'accès des .nc de modèle
    modele: 'arome' ou 'arpege'
    p: paramètre d'interpolation spatiale
    df_distance :  dataframe contenant les distances avec les K voisins les plus proches
    df_X: dataframe à compléter (test)
    var: string nom de la variable
    
    outputs:
    df_X: dataframe complété
    '''
    jour = df_X[df_X["forecast_"+model+"_"+var].isnull()]["date"].unique()


    for j in jour : 
        print(j)
        date_before = j-1
        date_after = j+1
        
        bool_before = True
        while bool_before :
            try :
                data_before = chargement_data_test(date_before, path, model)
            except : 
                date_before -= 1 
            else :
                bool_before = False
                print("date avant :", date_before)
                
                
        bool_after = True
        while bool_after :
            try :
                data_after = chargement_data_test(date_after, path, model)
            except : 
                date_after += 1 
            else :
                bool_after = False
                print("date après :", date_after)
        

        for nb in df_X[df_X['date']==j]['number_sta'] : 
            ind = df_distance[df_distance["number_sta"]== nb]["ind"].values[0]
            dist_min = df_distance[df_distance["number_sta"]== nb]["dist_min"].values[0]
            precip_before = prevision(ind, dist_min, data_before, p,var)
            precip_after = prevision(ind, dist_min, data_after, p,var)
            prev = np.mean([precip_before,precip_after])
           
        
            df_X[(df_X["date"]==j)& (df_X["number_sta"]==nb)]["forecast_"+model+"_"+var]  = prev
        
    return df_X
                

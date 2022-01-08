def X_test_preprocessing(df):
    
    # Récupération N° Station/jour/heure par ligne
    hour = df["Id"].str.split("_", n = 2, expand = True)[2]
    df['hour'] = hour.astype(int)
    day = df["Id"].str.split("_", n = 2, expand = True)[1]
    df['day'] = day.astype(int)
    nb_station = df["Id"].str.split("_", n = 2, expand = True)[0]
    df['number_sta'] = nb_station.astype(int)
    
    # Tri des features : N° Station / Jour / Heure
    df = df.sort_values(["number_sta","day",'hour'])
    df['Id'] = df['number_sta'].astype(str).str.cat(day,sep="_")
    df = df.drop(['hour','day'],axis=1) 
    
    return df
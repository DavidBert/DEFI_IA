import pandas as pd
def X_train_preprocessing(df):
    
    # Récupération mois/jour/heure par ligne
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    hour = df["Id"].str.split("_", n = 2, expand = True)[2]
    df['hour'] = hour.astype(int)
    day = df["Id"].str.split("_", n = 2, expand = True)[1]
    df['day'] = day.astype(int)
    
    # Création d'un Id quotidien pour pouvoir merge X_train avec Y_train
    df['Id_merge'] = df['number_sta'].astype(str).str.cat(day,sep="_")
    
    # Mise en ordre des features
    df = df[['dd','hu','td','t','ff','precip','month','Id','Id_merge','number_sta','hour','day']]
    df['precip'] = df['precip']*24
    df['month'] = df['month'].astype(int)
    
    # Tri des features : N° Station / Jour / Heure
    df = df.sort_values(["number_sta","day",'hour'])
    df = df.drop(['hour','day'],axis=1)
    
    return df
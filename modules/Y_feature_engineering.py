def Y_preprocessing(df):
    
    df = df.drop(['date','number_sta'],axis=1)
    df = df[['Id','Ground_truth']]
    df['Id_merge'] = df['Id']
    df = df.dropna()
    
    return df
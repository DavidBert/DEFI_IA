def Trainset_creation(X_train_df,Y_train_df):
    trainset = X_train_df.merge(Y_train_df,how="inner",on="Id_merge")
    trainset['Id'] = trainset['Id_merge']
    trainset = trainset.drop(['Id_x','Id_merge','Id_y'],axis=1)
    trainset = trainset.groupby("Id").mean()
    trainset = trainset.reset_index()
    trainset['month'] = trainset['month'].astype(int)
    return trainset
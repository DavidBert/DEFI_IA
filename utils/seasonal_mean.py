import os
import warnings

import pandas as pd

warnings.simplefilter(action='ignore')


# -----------------------------------------------------
#    Functions used to compute seasonal means.
#    Fill Nans with seasonal means
# -----------------------------------------------------

def get_season(row):
    """
    Get the season for each row of a dataframe.
    Create a 'season' column in dataframe.

    """
    m = row['month']
    if m in [1, 2, 12]:
        season = 'djf'
    if m in [3, 4, 5]:
        season = 'mam'
    if m in [6, 7, 8]:
        season = 'jja'
    if m in [9, 10, 11]:
        season = 'son'
    return season


def seasonal_mean(X, param):
    """
    Compute the 4 seasonal means of a parameter in the dataframe X.

    Args:
        X (dataframe)
        param (string): parameter in dataframe
    Returns:
        seasonal_mean (list) : seasonal means for 'djf', 'jja', 'mam', 'son'
    """

    X_new = X[['Id', 'month', 'season', param]]
    X_new['season'] = X_new.apply(get_season, axis=1)
    groups = X_new.groupby('season')
    seasonal_mean = [g[1][param].mean(skipna=True) for g in groups]
    return seasonal_mean


def fillna_with_seasonal_mean(X, param):
    """
    Fill missing values of a parameter with its seasonal means.

    Args:
        X (dataframe)
        param (string): parameter in dataframe
    Returns:
        X_new : the filled dataframe
    """

    Xnew = X.copy()
    Na = Xnew.index[Xnew[param].isna()]
    season = []

    for m in Xnew.month.values:
        if m in [1, 2, 12]:
            season += ['djf']
        if m in [3, 4, 5]:
            season += ['mam']
        if m in [6, 7, 8]:
            season += ['jja']
        if m in [9, 10, 11]:
            season += ['son']

    Xnew['season'] = season
    mean = seasonal_mean(Xnew, param)
    groups = Xnew.groupby('season')
    for group in groups:
        seas = group[0]
        ids = group[1].Id.values
        if seas == 'djf':
            Xnew[param][(Xnew.Id.isin(ids)) & (X.index.isin(Na))] = mean[0]
        elif seas == 'jja':
            Xnew[param][(Xnew.Id.isin(ids)) & (X.index.isin(Na))] = mean[1]
        elif seas == 'mam':
            Xnew[param][(Xnew.Id.isin(ids)) & (X.index.isin(Na))] = mean[2]
        else:
            Xnew[param][(Xnew.Id.isin(ids)) & (X.index.isin(Na))] = mean[3]
    return Xnew


# ----------------------------------------------------------

def fill_data(path, filename, params):
    X = pd.read_csv(os.path.join(path, filename), header=0, sep=',')
    for param in params:
        X = fillna_with_seasonal_mean(X, param)
    X.drop(columns='season', inplace=True)
    X.to_csv(os.path.join(path, filename), index=False)


def main():
    trainpath = os.path.join("Data", "Train", "Train")
    testpath = os.path.join("Data", "Test", "Test")

    params = [p + '_' + i for p in ['ws', 'p3031', 't2m', 'd2m', 'u10',
                                    'v10', 'r', 'tp', 'msl']
              for i in ['mean_day', 'std_day']]
    fill_data(trainpath, 'arpege2D.csv', params)
    fill_data(testpath, 'arpege2D.csv', params)

    params = [p + '_' + i for p in ['p3014_850hPa', 't_500hPa', 'z_500hPa',
                                    'r_700hPa', 'ws_1000hPa', 'w_950hPa']
              for i in ['mean_day', 'std_day']]
    fill_data(trainpath, 'arpege3D.csv', params)
    fill_data(testpath, 'arpege3D.csv', params)

    params = ['baseline_forecast']
    fill_data(trainpath, 'X_train.csv', params)
    fill_data(testpath, 'X_test.csv', params)


if __name__ == "__main__":
    main()

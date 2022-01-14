import os
import warnings

import pandas as pd

warnings.simplefilter(action='ignore')


def daily_mean(df, param, valid_times):
    df_sub = df[[param + '_' + str(i) for i in valid_times]]
    return df_sub.mean(axis=1, skipna=True).values


def daily_std(df, param, valid_times):
    df_sub = df[[param + '_' + str(i) for i in valid_times]]
    return df_sub.std(axis=1, skipna=True).values


def add_daily_mean(X, params, valid_times):
    for param in params:
        X[param + '_mean_day'] = daily_mean(X, param, valid_times)
    return X


def add_daily_std(X, params, valid_times):
    for param in params:
        X[param + '_std_day'] = daily_std(X, param, valid_times)
    return X


def daily_data(path, filename, params, valid_times):
    X = pd.read_csv(os.path.join(path, filename), header=0, sep=',')
    X = add_daily_mean(X, params, valid_times)
    X = add_daily_std(X, params, valid_times)
    X.drop(columns=[p + '_' + str(i) for i in valid_times for p in params],
           inplace=True)

    X.to_csv(os.path.join(path, filename), index=False)


def main():
    trainpath = os.path.join("Data", "Train", "Train")
    testpath = os.path.join("Data", "Test", "Test")

    params = ['ff', 't', 'td', 'hu', 'dd', 'precip']
    daily_data(trainpath, 'X_train.csv', params, range(24))
    daily_data(testpath, 'X_test.csv', params, range(24))

    params = ['ws', 'p3031', 't2m', 'd2m', 'u10', 'v10', 'r', 'tp', 'msl']
    daily_data(trainpath, 'arpege2D.csv', params, range(24))
    daily_data(testpath, 'arpege2D.csv', params, range(24))

    params = ['p3014_850hPa', 'z_500hPa', 't_500hPa',
              'r_700hPa', 'ws_1000hPa', 'w_950hPa']
    daily_data(trainpath, 'arpege3D.csv', params, range(17))
    daily_data(testpath, 'arpege3D.csv', params, range(17))


if __name__ == "__main__":
    main()

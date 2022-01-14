import os
import warnings

import pandas as pd

warnings.simplefilter(action='ignore')


# --------------------------------------------
#   Functions to read csv
#
# --------------------------------------------

def split_date(data):

    # date must be str or datetime

    year, month, day, hour = [], [], [], []
    for date in data.date.values:
        d = date.split('-')
        year += [d[0]]
        month += [d[1]]
        day += [d[2].split(' ')[0]]
        hour += [d[2].split(' ')[1].split(':')[0]]
    data['year'] = year
    data['month'] = month
    data['day'] = day
    data['hour'] = hour
    return data


def split_Id(data, Id_month_test):
    # date must be str or datetime
    number_sta, month, hour, index_group = [], [], [], []
    j = 0
    for ind in data.Id.values:
        d = ind.split('_')
        number_sta += [d[0]]
        day_index = d[1]
        m = Id_month_test[Id_month_test.day_index ==
                          int(day_index)].month.values
        month += [int(m)]
        hour += [int(d[2])]
        index_group += [d[0]+'_'+d[1]]
        j += 1
    data['month'] = month
    data['number_sta'] = number_sta
    data['hour'] = hour
    data['numbersta_day'] = index_group
    return data


def read_csv(df_path, date_parser=['date'], **kwargs):
    df = pd.read_csv(df_path, date_parser=date_parser, **kwargs)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    if 'Y_train' not in df_path:
        df['hour'] = df['date'].dt.hour
    return df


def add_baseline(X, baseline_obs, baseline_forecast):
    baseline_forecast = pd.read_csv(baseline_forecast, header=0, sep=',')
    baseline_obs = pd.read_csv(baseline_obs, header=0, sep=',')
    X = X.merge(baseline_obs[['Prediction', 'Id']], on=['Id'], how='left')
    X.rename(columns={'Prediction': 'baseline_obs'}, inplace=True)
    X = X.merge(baseline_forecast[['Prediction', 'Id']], on=['Id'], how='left')
    X.rename(columns={'Prediction': 'baseline_forecast'}, inplace=True)
    return X


# --------------------------------------------------------------------------------------------

# TRAIN

def create_X_stations_train(X_station_train, trainpath):
    X = X_station_train.groupby(['number_sta', 'year', 'month', 'day'])
    for group in X:
        if group[1].shape[0] != 24:
            gid = group[0]
            id_group = group[1].Id.values[0]
            for h in range(24):
                if sum(group[1].hour == h) == 0:
                    new_row = {
                        'number_sta': int(gid[0]),
                        'year': int(gid[1]),
                        'month': int(gid[2]),
                        'day': int(gid[3]),
                        'hour': int(h),
                        'Id': '_'.join(str(id_group).split('_')[:2])
                        + '_'
                        + str(h),
                    }

                    X_station_train = X_station_train.append(
                        new_row, ignore_index=True)

    X_station_train['Id'] = ['_'.join(str(i).split(
        '_')[:2]) for i in X_station_train['Id'].values]
    for h in range(24):
        group = X_station_train[X_station_train['hour'] == h].reset_index()
        group.rename(columns={'ff': 'ff_'+str(h), 't': 't_'+str(h), 'td': 'td_'+str(
            h), 'hu': 'hu_'+str(h), 'dd': 'dd_'+str(h), 'precip': 'precip_'+str(h)}, inplace=True)
        if h == 0:
            X = group
        else:
            X = X.merge(group[['ff_'+str(h), 't_'+str(h), 'td_'+str(h), 'hu_'+str(h),
                               'dd_'+str(h), 'precip_'+str(h), 'Id']], on=['Id'], how='left')
    X.drop(columns=['hour', 'index'], inplace=True)
    X = add_baseline(X, os.path.join(trainpath, 'Baselines', 'Baseline_observation_train.csv'),
                     os.path.join(trainpath, 'Baselines', 'Baseline_forecast_train.csv'))
    indexes = X.index[X.baseline_obs.isna()]
    X.drop(index=indexes, inplace=True)

    # Save data
    X.to_csv(os.path.join(trainpath, 'X_train.csv'), index=False)


# TEST

def create_X_stations_test(X_station_test, testpath):
    Id_month_test = pd.read_csv(os.path.join(
        testpath, 'Id_month_test.csv'), header=0, sep=',')
    X_station_test = split_Id(X_station_test, Id_month_test)
    X = X_station_test.groupby(['numbersta_day'])
    for group in X:
        if group[1].shape[0] != 24:
            gid = group[0]
            id_group = group[1].Id.values[0]
            for h in range(24):
                if sum(group[1].hour == h) == 0:
                    m = Id_month_test[Id_month_test.day_index == int(
                        gid[1])].month.values
                    new_row = {
                        'number_sta': int(gid[0]),
                        'month': int(m),
                        'hour': int(h),
                        'Id': '_'.join(str(id_group).split('_')[:2])
                        + '_'
                        + str(h),
                        'numbersta_day': gid,
                    }

                    X_station_test = X_station_test.append(
                        new_row, ignore_index=True)
    X_station_test['Id'] = [
        '_'.join(str(i).split('_')[:2]) for i in X_station_test['Id'].values
    ]

    for h in range(24):
        group = X_station_test[X_station_test['hour'] == h].reset_index()
        group.rename(columns={'ff': 'ff_'+str(h), 't': 't_'+str(h), 'td': 'td_'+str(
            h), 'hu': 'hu_'+str(h), 'dd': 'dd_'+str(h), 'precip': 'precip_'+str(h)}, inplace=True)
        if h == 0:
            X = group
        else:
            X = X.merge(group[['ff_'+str(h), 't_'+str(h), 'td_'+str(h), 'hu_'+str(h),
                               'dd_'+str(h), 'precip_'+str(h), 'Id']], on=['Id'], how='left')
    X.drop(columns=['hour', 'index'], inplace=True)
    X = add_baseline(X, os.path.join(testpath, 'Baselines', 'Baseline_observation_test.csv'),
                     os.path.join(testpath, 'Baselines', 'Baseline_forecast_test.csv'))
    indexes = X.index[X.baseline_obs.isna()]
    X.drop(index=indexes, inplace=True)

    # Save data
    X.to_csv(os.path.join(testpath, 'X_test.csv'), index=False)


def main():
    trainpath = os.path.join("Data", "Train", "Train")
    X_train = read_csv(os.path.join(
        trainpath, 'X_station_train.csv'), date_parser=['date'])
    create_X_stations_train(X_train, trainpath)

    testpath = os.path.join("Data", "Test", "Test")
    X_test = pd.read_csv(os.path.join(
        testpath, 'X_station_test.csv'), header=0, sep=',')
    create_X_stations_test(X_test, testpath)


if __name__ == "__main__":
    main()

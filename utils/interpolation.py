import os
import warnings

import numpy as np
import pandas as pd
from haversine import haversine

warnings.simplefilter(action='ignore')


# -----------------------------------------------------
#    Functions used to spatially interpolate stations
#    Fill Nans by spatial interpolation.
# -----------------------------------------------------


def get_distance(number_sta1, number_sta2, coords):
    """
    Compute the distance between two stations.

    Args:
        number_sta1 and number_sta2: number of the 2 stations
    Returns:
        distance (float): distance between the 2 stations
    """
    station1 = coords[['lat', 'lon']][coords.number_sta == number_sta1]
    station2 = coords[['lat', 'lon']][coords.number_sta == number_sta2]
    station1 = (float(station1.lat.values), float(station1.lon.values))
    station2 = (float(station2.lat.values), float(station2.lon.values))
    distance = haversine(station1, station2)
    return distance


def get_closest_stations(coords, number_sta, k):
    """
    Get the k closest stations.

    Args:
        number_sta : number of the station
        k (int) : number of neighbours
    Returns:
        dfk (dataframe): number of the k closest stations and their distance.
    """
    stations = coords.drop(index=coords[coords.number_sta == number_sta].index)
    number_stations = np.unique(stations.number_sta.values)

    dist = []
    for n in number_stations:
        dist += [get_distance(number_sta, n, coords)]
    df = pd.DataFrame(data={'number_sta': number_stations, 'distance': dist}).sort_values(
        by='distance', ascending=True)
    dfk = df.head(k).reset_index().drop(columns='index')
    return dfk


def weighted_mean(X, param, df):
    """
    Compute the mean of a param for k stations weigthed by their distance to the station to fill.

    Args:
        X (dataframe) : the data
        param (string) : the variable from X to interpolate
        df (dataframe) : number of the k stations and their distance
    Returns:
        the weigthed mean (float)

    """
    dist = df.distance.values
    sta = df.number_sta.values
    ci, somme = 0, 0
    for k in range(df.shape[0]):
        if not(np.isnan(X[X.number_sta == sta[k]][param].values)) and len(X[X.number_sta == sta[k]][param].values) >= 1:
            ci += 1/dist[k] * float(X[X.number_sta == sta[k]][param].values)
            somme += 1/dist[k]
    if somme == 0:
        return np.nan
    return ci/somme


def interpolation_spatiale(X, coords, param, k):
    """
    Impute missing values in the column 'param' of X,
    by weighted mean between the k closest stations,
    at the same timestamp.

    Args:
        X (dataframe) : the data
        param (string) : the variable from X to interpolate
        k(int) : number of neighbours
    Returns:
        X_new : the filled dataframe

    """
    X_new = X.copy()

    # Get the Id where param is missing
    sta_to_fill = X_new.Id[X_new[param].isna()].values
    j = 0
    for sta in sta_to_fill:
        number_sta = int(sta.split('_')[0])
        day = sta.split('_')[1]

        # Get closest stations
        df = get_closest_stations(coords, number_sta, k)

        # Get the Id of these stations for the good day
        id_sta = np.array([str(s)+'_'+day for s in df.number_sta.values])

        # Compute weigthed mean
        X_sub = X[X.Id.isin(id_sta)]
        X_new[param][X_new.Id == sta] = weighted_mean(X_sub, param, df)
        j += 1
    return X_new


def interpolate_all_params(X, path, filename, coords):
    X_interpolate = X.copy()
    params = [i+str(j) for i in ['ff', 't', 'td', 'precip', 'hu', 'dd']
              for j in ['_mean_day', '_std_day']]
    for param in params:
        X_interpolate = interpolation_spatiale(X, coords, param, 5)

    X_interpolate.to_csv(os.path.join(path, filename), header=0, sep=',')


def main():
    otherpath = os.path.join("Data", "Other", "Other")
    trainpath = os.path.join("Data", "Train", "Train")
    testpath = os.path.join("Data", "Test", "Test")

    X_train = pd.read_csv(os.path.join(
        trainpath, 'X_train.csv'), header=0, sep=',')
    coords = pd.read_csv(os.path.join(
        otherpath, 'stations_coordinates.csv'), header=0, sep=',')
    coords = coords[coords.number_sta.isin(X_train.number_sta)]
    interpolate_all_params(X_train, trainpath, 'X_train.csv', coords)

    X_test = pd.read_csv(os.path.join(
        testpath, 'X_test.csv'), header=0, sep=',')
    coords = pd.read_csv(os.path.join(
        otherpath, 'stations_coordinates.csv'), header=0, sep=',')
    coords = coords[coords.number_sta.isin(X_test.number_sta)]
    interpolate_all_params(X_test, testpath, 'X_test.csv', coords)


if __name__ == "__main__":
    main()

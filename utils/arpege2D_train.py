import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from haversine import haversine

warnings.simplefilter(action='ignore')


# --------------------------------------------
#	Functions to get ARPGE 2D
#
# --------------------------------------------


def get_grid(filenc):
    """
    Get the coordinates of ARPEGE grid points.
    """

    ar = xr.open_dataset(filenc, decode_times=True)
    ar.close()
    latitudes = ar['latitude'].values
    longitudes = ar['longitude'].values
    return latitudes, longitudes


def closest_grid_point(coords, station, latitudes, longitudes):
    """
    Computes the distance between the station and each grid point.
    Returns : (lat,lon) the coordinates of the grid point closest to the station.
    """
    lat_station = coords.lat[coords.number_sta == station]
    lon_station = coords.lon[coords.number_sta == station]
    station = (float(lat_station.values), float(lon_station.values))
    grid, distance = [], []
    for lat in latitudes:
        for lon in longitudes:
            grid_point = (lat, lon)
            grid += [grid_point]
            distance += [haversine(station, grid_point)]
    return grid[np.argmin(distance)]


def create_forecast_id(row):
    Id = row['Id']
    station = Id.split('_')[0]
    day = Id.split('_')[1]
    forecast_day = int(day) - 1
    return station+str('_')+str(forecast_day)


def create_filename(row):
    year = row['year']
    month = row['month']
    day = row['day']
    if int(day) < 10:
        day = '0'+str(day)
    if int(month) < 10:
        month = '0'+str(month)
    return '2D_arpege_'+str(year)+str(month)+str(day)+'.nc'


def create_foldername(row):
    year = row['year']
    return '2D_arpege_'+str(year)


def param_id_valid_times(params):
    return [p+str('_')+str(t) for p in params for t in range(25)]


def extract_param(row):
    trainpath = os.path.join("Data", "Train", "Train")
    folder = row['folder']
    filenc = row['filename']
    grid_lat = row['grid_lat']
    grid_lon = row['grid_lon']
    n_valid_times = 25

    params = ['ws', 'p3031', 't2m', 'd2m', 'u10', 'v10', 'r', 'tp', 'msl']
    columns = np.zeros(len(params)*n_valid_times)
    columns[:] = np.nan

    # If file is missing
    if not(os.path.isfile(os.path.join(trainpath, 'X_forecast', folder, filenc))):
        print("Missing file:", filenc)
        return columns

    # Select file to open
    ar = xr.open_dataset(os.path.join(
        trainpath, 'X_forecast', folder, filenc), decode_times=True)
    ar.close()

    # Select data at closest grid point
    data = ar.sel(latitude=grid_lat, longitude=grid_lon)

    j = 0
    for p in params:
        p_valid_times = data[p].values

        # Check params have the correct length
        if len(p_valid_times) != n_valid_times:
            print('Erreur lenght param:', p, filenc)

        columns[j*n_valid_times:(j+1)*n_valid_times] = p_valid_times
        j += 1
    return columns


def split_date(row):
    date = row['date']
    day = date.dt.day
    month = date.dt.month
    year = date.dt.year
    return [day, month, year]


def DataFrame_grid_points(number_sta, latitudes, longitudes, coords):
    grid_lat, grid_lon = [], []
    for sta in np.unique(number_sta):
        grid_point = closest_grid_point(coords, sta, latitudes, longitudes)
        grid_lat += [grid_point[0]]
        grid_lon += [grid_point[1]]
    data = pd.DataFrame({'number_sta': np.unique(
        number_sta), 'grid_lat': grid_lat, 'grid_lon': grid_lon})
    return data


# -----------------------------------------------


def arpege2D_train(X_train, trainpath, otherpath):
    # Import stations
    coords = pd.read_csv(os.path.join(
        otherpath, 'stations_coordinates.csv'), header=0, sep=',')

    # Get station informations
    arpege2D = X_train[['number_sta', 'date', 'Id']]

    # Get neighbours for each station
    filenc = os.path.join(trainpath, 'X_forecast',
                          '2D_arpege_2016', '2D_arpege_20160102.nc')
    latitudes, longitudes = get_grid(filenc)
    grid_points_coords = DataFrame_grid_points(
        X_train.number_sta.values, latitudes, longitudes, coords)
    arpege2D = arpege2D.merge(grid_points_coords, on=[
                              'number_sta'], how='left')

    # Localize Nan in date
    NaIndex = arpege2D.index[arpege2D.date.isna()]
    arpege2D.drop(index=NaIndex, inplace=True)

    # Create a forecast id
    arpege2D['date'] = pd.to_datetime(arpege2D['date'])
    arpege2D['day'] = arpege2D['date'].dt.day
    arpege2D['month'] = arpege2D['date'].dt.month
    arpege2D['year'] = arpege2D['date'].dt.year
    arpege2D['filename'] = arpege2D.apply(create_filename, axis=1)
    arpege2D['folder'] = arpege2D.apply(create_foldername, axis=1)
    arpege2D = arpege2D.sort_values(
        ['number_sta', 'day', 'date']).reset_index(drop=True)
    arpege2D['forecast_id'] = arpege2D.apply(create_forecast_id, axis=1)

    # Extract Params
    params = ['ws', 'p3031', 't2m', 'd2m', 'u10', 'v10', 'r', 'tp', 'msl']
    columns = param_id_valid_times(params)
    arpege2D[columns] = arpege2D.apply(
        extract_param, axis=1, result_type='expand')

    # Rename columns
    params = ['ws', 'p3031', 't2m', 'd2m', 'u10', 'v10', 'r', 'msl']
    arpege2D.drop(columns=[p+'_24' for p in params], inplace=True)
    arpege2D.drop(columns='tp_0', inplace=True)
    arpege2D.drop(columns=['folder', 'filename', 'year',
                           'month', 'day', 'grid_lon', 'grid_lat'], inplace=True)
    arpege2D.dropna(inplace=True)
    arpege2D.drop(columns='Id', inplace=True)
    arpege2D.rename(columns={'forecast_id': 'Id'}, inplace=True)
    for i in range(24):
        arpege2D.rename(columns={'tp_'+str(i+1): 'tp_'+str(i)}, inplace=True)

    # Save data
    arpege2D.to_csv(os.path.join(trainpath, 'arpege2D.csv'), index=False)


def main():
    trainpath = os.path.join("Data", "Train", "Train")
    otherpath = os.path.join("Data", "Other", "Other")
    X_train = pd.read_csv(os.path.join(
        trainpath, 'X_station_train.csv'), header=0, sep=',')
    arpege2D_train(X_train, trainpath, otherpath)


if __name__ == "__main__":
    main()

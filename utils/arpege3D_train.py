import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from haversine import haversine

warnings.simplefilter(action='ignore')


# --------------------------------------------
#	Functions to get ARPGE 3D
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
    closest_point = grid[np.argmin(distance)]
    return closest_point


def create_forecast_id(row):
    Id = row['Id']
    station = Id.split('_')[0]
    day = Id.split('_')[1]
    forecast_day = int(day) - 1
    new_id = station+str('_')+str(forecast_day)
    return new_id


def create_filename(row):
    year = row['year']
    month = row['month']
    day = row['day']
    if int(day) < 10:
        day = '0'+str(day)
    if int(month) < 10:
        month = '0'+str(month)
    filename = 'arpege_3D_isobar_'+str(year)+str(month)+str(day)+'.nc'
    return filename


def create_foldername(row):
    year = row['year']
    folder = '3D_arpege_'+str(year)
    return folder


def param_id_valid_times(params):
    columns = [p+str('_')+str(t) for p in params for t in range(17)]
    return columns


def extract_param(row):
    trainpath = os.path.join("Data", "Train", "Train")
    folder = row['folder']
    filenc = row['filename']
    grid_lat = row['grid_lat']
    grid_lon = row['grid_lon']
    n_valid_times = 17

    params = ['p3014', 'z', 't', 'r', 'ws', 'w']
    isobarics = [850, 500, 500, 700, 1000, 950]

    columns = np.zeros(len(params)*n_valid_times)
    columns[:] = np.nan

    # If file is missing
    if not(os.path.isfile(os.path.join(trainpath, 'X_forecast', folder, filenc))):
        print(filenc)
        return columns

    # Select file to open
    ar = xr.open_dataset(os.path.join(
        trainpath, 'X_forecast', folder, filenc), decode_times=True)
    ar.close()

    # Get params for all valid_times

    for i in range(len(params)):

        try:
            # Select data at closest grid point
            data = ar.sel(latitude=grid_lat, longitude=grid_lon,
                          isobaricInhPa=isobarics[i])

            p = params[i]
            p_valid_times = data[p].values

            columns[i*n_valid_times:(i+1)*n_valid_times] = p_valid_times

        except:
            # Print which file is not valid
            print(filenc)

    return columns


def split_date(row):
    date = row['date']
    day = date.dt.day
    month = date.dt.month
    year = date.dt.year
    split = [day, month, year]
    return split


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


def arpege3D_train(X_train, trainpath, otherpath):

    # Import stations
    coords = pd.read_csv(os.path.join(
        otherpath, 'stations_coordinates.csv'), header=0, sep=',')

    # Get station informations
    arpege3D = X_train[['number_sta', 'date', 'Id']]

    # Get neighbours for each station
    filenc = os.path.join(trainpath, 'X_forecast',
                          '3D_arpege_2016', 'arpege_3D_isobar_20160102.nc')
    latitudes, longitudes = get_grid(filenc)
    grid_points_coords = DataFrame_grid_points(
        X_train.number_sta.values, latitudes, longitudes, coords)
    arpege3D = arpege3D.merge(grid_points_coords, on=[
        'number_sta'], how='left')

    # Localize Nan in date
    NaIndex = arpege3D.index[arpege3D.date.isna()]
    arpege3D.drop(index=NaIndex, inplace=True)

    # Create a forecast id
    arpege3D['date'] = pd.to_datetime(arpege3D['date'])
    arpege3D['day'] = arpege3D['date'].dt.day
    arpege3D['month'] = arpege3D['date'].dt.month
    arpege3D['year'] = arpege3D['date'].dt.year
    arpege3D['filename'] = arpege3D.apply(create_filename, axis=1)
    arpege3D['folder'] = arpege3D.apply(create_foldername, axis=1)
    arpege3D = arpege3D.sort_values(
        ['number_sta', 'day', 'date']).reset_index(drop=True)
    arpege3D['forecast_id'] = arpege3D.apply(create_forecast_id, axis=1)

    # Extract Params
    params = ['p3014_850hPa', 'z_500hPa', 't_500hPa',
              'r_700hPa', 'ws_1000hPa', 'w_950hPa']
    columns = param_id_valid_times(params)
    arpege3D[columns] = arpege3D.apply(
        extract_param, axis=1, result_type='expand')

    # Rename columns
    arpege3D.drop(columns=['folder', 'filename', 'year',
                           'month', 'day', 'grid_lon', 'grid_lat'], inplace=True)
    arpege3D.dropna(inplace=True)
    arpege3D.drop(columns='Id', inplace=True)
    arpege3D.rename(columns={'forecast_id': 'Id'}, inplace=True)

    # Save data
    arpege3D.to_csv(os.path.join(trainpath, 'arpege3D.csv'), index=False)


def main():
    trainpath = os.path.join("Data", "Train", "Train")
    otherpath = os.path.join("Data", "Other", "Other")
    X_train = pd.read_csv(os.path.join(
        trainpath, 'X_station_train.csv'), header=0, sep=',')
    arpege3D_train(X_train, trainpath, otherpath)


if __name__ == "__main__":
    main()

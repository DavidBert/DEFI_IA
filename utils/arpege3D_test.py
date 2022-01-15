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


def create_filename(row):
    Id = row['Id']
    day = Id.split('_')[1]
    filename = 'arpege_3D_isobar_'+str(day)+'.nc'
    return filename


def param_id_valid_times(params):
    columns = [p+str('_')+str(t) for p in params for t in range(17)]
    return columns


def extract_param(row):
    testpath = os.path.join("Data", "Test", "Test")
    folder = '3D_arpege'
    filenc = row['filename']
    grid_lat = row['grid_lat']
    grid_lon = row['grid_lon']
    n_valid_times = 17

    params = ['p3014', 'z', 't', 'r', 'ws', 'w']
    isobarics = [850, 500, 500, 700, 1000, 950]
    columns = np.zeros(len(params)*n_valid_times)
    columns[:] = np.nan

    # If file is missing
    if not(os.path.isfile(os.path.join(testpath, 'X_forecast', folder, filenc))):
        print(filenc)
        return columns

    # Select file to open
    ar = xr.open_dataset(os.path.join(
        testpath, 'X_forecast', folder, filenc), decode_times=True)
    ar.close()

    # Select data at closest grid point
    data = ar.sel(latitude=grid_lat, longitude=grid_lon)

    # Get params for all valid_times
    j = 0

    for i in range(len(params)):
        try:
            # Select data at closest grid point
            data = ar.sel(latitude=grid_lat, longitude=grid_lon,
                          isobaricInhPa=isobarics[i])

            p = params[i]
            p_valid_times = data[p].values

            columns[j*n_valid_times:(j+1)*n_valid_times] = p_valid_times
        except:
            print(filenc)
        j += 1

    return columns


def DataFrame_grid_points(number_sta, latitudes, longitudes, coords):
    grid_lat, grid_lon = [], []
    for sta in np.unique(number_sta):
        grid_point = closest_grid_point(coords, sta, latitudes, longitudes)
        grid_lat += [grid_point[0]]
        grid_lon += [grid_point[1]]
    data = pd.DataFrame({'number_sta': np.unique(
        number_sta), 'grid_lat': grid_lat, 'grid_lon': grid_lon})
    return data

# ------------------------------------------------------


def arpege3D_test(X_test, testpath, otherpath):

    # Import stations
    coords = pd.read_csv(os.path.join(
        otherpath, 'stations_coordinates.csv'), header=0, sep=',')

    # Get station informations
    arpege3D = X_test[['number_sta', 'Id']]

    # Get neighbours for each station
    filenc = os.path.join(testpath, 'X_forecast',
                          '3D_arpege', 'arpege_3D_isobar_0.nc')
    latitudes, longitudes = get_grid(filenc)
    grid_points_coords = DataFrame_grid_points(
        X_test.number_sta.values, latitudes, longitudes, coords)
    arpege3D = arpege3D.merge(grid_points_coords, on=[
                              'number_sta'], how='left')
    arpege3D['filename'] = arpege3D.apply(create_filename, axis=1)

    # Extract Params
    params = ['p3014_850hPa', 'z_500hPa', 't_500hPa',
              'r_700hPa', 'ws_1000hPa', 'w_950hPa']
    columns = param_id_valid_times(params)
    arpege3D[columns] = arpege3D.apply(
        extract_param, axis=1, result_type='expand')

    # drop columns
    arpege3D.drop(columns=['filename', 'grid_lon', 'grid_lat'], inplace=True)

    # Save data
    arpege3D.to_csv(os.path.join(testpath, 'arpege3D.csv'), index=False)


def main():
    testpath = os.path.join("Data", "Test", "Test")
    otherpath = os.path.join("Data", "Other", "Other")
    X_test = pd.read_csv(os.path.join(
        testpath, 'X_station_test.csv'), header=0, sep=',')
    arpege3D_test(X_test, testpath, otherpath)


if __name__ == "__main__":
    main()

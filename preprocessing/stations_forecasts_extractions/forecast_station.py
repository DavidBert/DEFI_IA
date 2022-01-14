from math import radians, cos, sin, asin, sqrt
from datetime import datetime, timedelta
import csv 
import sys
import shutil
import numpy as np

DISTANCE = 10

def qick_distance(Lat1, Long1, Lat2, Long2):
    x = Lat2 - Lat1
    y = (Long2 - Long1)*cos((Lat2 + Lat1)*0.00872664626)  
    return 111.138*sqrt(x*x+y*y)


def print_msg(*args,end='\n'):
    for item in args:
        sys.stdout.write(str(item)+' ')
    sys.stdout.write(end)
    sys.stdout.flush()

print(qick_distance(43.5405365,1.5128983, 44.0216604,1.2945122))

#Read Stations positions 
stations_positions_file = open('defi-ia-2022/Other/Other/stations_coordinates.csv')
stations_positions_csvreader = csv.reader(stations_positions_file)

header = next(stations_positions_csvreader)

stations_positions = []
for row in stations_positions_csvreader:
    #print_msg(row)
    stations_positions.append(row)

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


start_date = datetime(2016, 7, 1)
end_date = datetime(2017, 1, 1)

result_file = open('results2016_2.csv', 'w', newline='')
writer = csv.writer(result_file)
for d in daterange(start_date, end_date):
    day_forecast_file = open('x_forcast/'+d.strftime("%Y%m%d")+'.csv')
    day_forecast_csvreader = csv.reader(day_forecast_file)
    # Switch header
    print_msg('day : '+ d.strftime("%Y%m%d"))
    print_msg('-------------------------------------------------------------------------')
    next(day_forecast_csvreader)
    forecasts = []
    for f in day_forecast_csvreader :
        forecasts.append(f)
    for station in stations_positions:
        rate = []
        weights = []
        has_neighbor = False
        for forecast in forecasts:
            dist = qick_distance(float(station[1]), float(station[2]), float(forecast[0]), float(forecast[1])) 
            if dist < DISTANCE : 
                rate.append(float(forecast[2]))
                weights.append(dist)
                has_neighbor = True
        if has_neighbor :
            writer.writerow([d.strftime("%Y-%m-%d"), station[0],np.average(rate, weights=weights)])
        else :
            writer.writerow([d.strftime("%Y-%m-%d"), station[0],0])
    day_forecast_file.close()

result_file.close()
stations_positions_file.close()
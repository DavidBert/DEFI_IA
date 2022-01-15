from netCDF4 import Dataset
from datetime import datetime, timedelta
import csv 
import sys
import shutil

def print_msg(*args,end='\n'):
    for item in args:
        sys.stdout.write(str(item)+' ')
    sys.stdout.write(end)
    sys.stdout.flush()

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

ds = Dataset("defi-ia-2022/Train/Train/X_forecast/2D_arome_20170214.nc", "r", format="NETCDF4")

f = open('latitudes_longitudes.csv', 'w', newline='')
writer = csv.writer(f)

header = ['latitude',  'longitude', 'tp']


latitudes = ds['latitude'][:]
longitudes = ds['longitude'][:]

# Starting date
start_date = datetime(2016, 1, 1)

# End date
end_date = datetime(2017, 1, 1)

for d in daterange(start_date, end_date):
    try:
        f = open('x_forcast/'+d.strftime("%Y%m%d")+'.csv', 'w', newline='')
        ds = Dataset("2D_arome_2016/2D_arome_"+d.strftime("%Y%m%d")+".nc", "r", format="NETCDF4")
        writer = csv.writer(f)
        writer.writerow(header)
        print_msg(d.strftime("%Y%m%d"))
        tp = ds['tp'][:]
        i = 0
        for lat in latitudes:
            j = 0
            for lon in longitudes :
                try:
                    val = tp[11,i,j]
                    writer.writerow([lat,lon,val])
                except:
                    print_msg('error in '+str(i)+ ' , '+str(j))
                j = j + 1
            i = i + 1
        f.close()
        ds.close()
        last_date = d
    except:
        print_msg('error in '+d.strftime("%Y%m%d"))
        shutil.copyfile('x_forcast/'+last_date.strftime("%Y%m%d")+'.csv', 'x_forcast/'+d.strftime("%Y%m%d")+'.csv')

   
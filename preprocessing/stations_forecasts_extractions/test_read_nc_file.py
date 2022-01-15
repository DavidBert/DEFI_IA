from netCDF4 import Dataset
from datetime import datetime

ds = Dataset("2D_arome_2016/2D_arome_20160812.nc", "r", format="NETCDF4")


print(ds)

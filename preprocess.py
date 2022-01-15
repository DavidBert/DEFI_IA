import os

from utils import (X_stations, arpege2D_test, arpege2D_train, arpege3D_test,
                   arpege3D_train, feature_engineering, interpolation,
                   merge_files, merge_y_id_month, seasonal_mean)
from utils.funcs import download_arpege

data_path = 'Data'

train_path = os.path.join(data_path, 'Train', 'Train')
test_path = os.path.join(data_path, 'Test', 'Test')
other_path = os.path.join(data_path, 'Other', 'Other')

for path in [train_path, test_path, other_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# Step 1: get data

# Retrieve downloaded data: X_station
print('Retrieving downloaded data: X_station')
X_stations.main()
# Download Arpege data
print('Downloading Arpege data')
download_arpege(data_path)

# # Open .nc files
print('Opening .nc files')
arpege2D_train.main()
arpege2D_test.main()
arpege3D_train.main()
arpege3D_test.main()

# # Step 2: feature engineering
print('Feature engineering')
feature_engineering.main()

# # Step 3: interpolation
print('Interpolation')
interpolation.main()

# # Step 4: fill NAs with seasonal mean
print('Fill NAs with seasonal mean')
seasonal_mean.main()

# # Step 5: merge files
print('Merge files')
merge_files.main()

# # Step 6: merge dataframes
print('Merge dataframes')
merge_y_id_month.main()

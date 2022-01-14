# Defi-IA-2022
Défi IA 2022 Kaggle - Local prediction of rainfall using data measure on ground stations and results from MétéoFrance predictive models (Arpège_2D). 

This code achieved a MAPE score of 26.57 (16th/84) on the public leaderboard and the competition results were a MAPE of 30.13, ranked 27th place (private leaderboard).

## You can skip to the training models part and directly downlaod final data

## Instructions to download the data, works on linux:
- Create your account on Kaggle
- Get your API credentials 

" To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. Place this file in the location ~/.kaggle/kaggle.json " and set its rwx rights to 600.
  
  - Be sure to have the `kaggle` command installed using `pip install kaggle`. If the kaggle command cannot be found, add `~/.local/bin` to your path.
  - Run `dowload_data.sh` in your working directory. A ./DATA_RAINFALL/ directory will be created containing all the data from ground stations and Arpege.
  
## Preprocessing the data:
  
  - Install the `xarray` python library (plus `netcdf4` and  `h5netcdf` if necessary) to collect Arpege_2D data
  - Open your terminal in your working directory and run the following command : `python preprocess_train.py your_working_directory_path`. Make sure to replace `your_working_directory_path` by your working directory path ! The script fills nans from X_station_train and Y_train, merges X_station_train with 2D_arpege_train and then reshapes the training features to hourly features (all features at each hour). Two files full_X_train.csv and full_Y_train.csv are created in the directory ./DATA_RAINFALL/Train/Train/. Preprocessing the training set is time-consumming (approximately 6 hours). 
   - Open your terminal in your working directory and run the following command : `python preprocess_test.py your_working_directory_path`. Make sure to replace `your_working_directory_path` by your working directory path ! The script fills nans in X_station_test, merges X_station_test with 2D_arpege_test and then reshapes the features to hourly features. A file full_X_test.csv is created in the directory ./DATA_RAINFALL/Test/Test/. Preprocessing the test set could last for around 30 minutes. Note that Y_test is not provided by MeteoFrance. 
  
## Training models and making predictions
  
  - **if you didnt run the preprocess part till the end :** Download the "Data.zip" file that contains the preprocessed data from : https://drive.google.com/file/d/10xF6B2JB-cEftuSWBBWLvF_FETbXBVpc/view?usp=sharing.
  Unzip the file, make sure the "Data" folder is next to `train.py`. This folder contains the data after preprocessing.
  
  -Run the following command, this runs the feature engineering before training on MLP and LGBM models, creating prediction .csv files afterwards in results. The training      takes 10 to 15 minutes.
  
  ```
  python train.py --data_path Data --output_folder Results
  
  ```
  

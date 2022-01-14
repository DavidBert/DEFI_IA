# Defi-IA-2022
Défi IA 2022 (Kaggle)

<a href="https://www.kaggle.com/c/defi-ia-2022" title = "Defi-IA 2022">
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/Kaggle_logo.png" width="120" alt="Kaggle">
</p>
</a>

<a href="http://www.insa-toulouse.fr/" title = "INSA Toulouse">
<p align="center">
  <img src="https://jpo.insa-toulouse.fr/wp-content/uploads/2020/11/Logo_INSAToulouse-quadri.png" width="200" alt="INSA Toulouse">
</p>
</a>

Team: STEC

Members: Célia Duluc, Thomas Nivelet, Elisa Escanez, Sébastien Castets

INSA Toulouse - 5th year (Applied Mathematics)

Objective: Predict the accumulated daily rainfall on the D day on the observation ground stations.


## Download data

- Option 1 : download all the data on your computer. Warning : data size of 80Gb and time consuming!
- Option 2 : download the final dataset with all the preprocessing already done. 

### Option 1
- Run the data_meteonet.sh file (in the data folder). It downloads forecast data.

```
sh data/data_meteonet.sh
```
- Go on Kaggle website -> profile -> Account -> Create New API Token
- Move the kaggle.json into the current directory
- Run the following command to download station data.

```
sh data/data_station.sh
```
### Option 2

- Download the csv files using the following commands
```
gdown --id 1UFRhVIOeXBG0N-5KFDnzRbrnmJhPbCKY -O data/df_train.csv 
gdown --id 1Qq20yHfeReGOVdv8XiriVu55EkI8I3rH -O data/df_X_test.csv
```

## Train the model and get the predictions

- Run the train.py, with the following arguments: 
  - data_path = './data', 
  - output_folder: an output folder, where you will get the predictions,
  - preprocessing:
    - *Option 1 (if you have downloaded all the data on your computer):*
      ```
      python train.py --data_path './data' --output_folder <OUTPUT_FOLDER> --preprocessing
      ```
    - *Option 2 (if you have downloaded the final dataset with the preprocessing already done):* 
      ```
      python train.py --data_path './data' --output_folder <OUTPUT_FOLDER> --no-preprocessing
      ```


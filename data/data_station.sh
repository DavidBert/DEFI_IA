# download X_station
mkdir -p data/data_station
mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c defi-ia-2022
unzip defi-ia-2022.zip -d data/data_station/
rm defi-ia-2022.zip


import argparse
import os
import warnings
import zipfile

import gdown
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

warnings.simplefilter(action='ignore')


def download_data(output_folder):
    urls = ["https://drive.google.com/uc?id=1YT8MByCh0svSvDTbg1k4ECL-WemnduoM",
            "https://drive.google.com/uc?id=1QmLUUfHwKedW7cVFpmUUcjM-BwG5Q2ku"]
    output_files = [os.path.join(output_folder, 'X_test_final.zip'),
                    os.path.join(output_folder, 'X_train_final.zip')]

    for url, output_file in zip(urls, output_files):
        if not os.path.exists(output_file):
            gdown.download(url, output_file, quiet=False)

            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                zip_ref.extractall(output_folder)
            os.remove(output_file)


def round_values(row):
    value = row['Prediction']
    return max(value, 1.0)


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='Folder to data files')
parser.add_argument(
    '--output_folder', help='Folder to save the model, training logs, and predictions')
args = parser.parse_args()

data_path = args.data_path
results_path = args.output_folder

for path in [data_path, results_path]:
    if not os.path.exists(path):
        os.makedirs(path)

download_data(data_path)

X = pd.read_csv(os.path.join(data_path, 'X_train_final.csv'),
                header=0, sep=',')
Y = pd.DataFrame({
    'Ground_truth': X['Ground_truth'].values,
    'Id': X['Id'].values
})
X = X.drop(columns=['date', 'Ground_truth', 'number_sta'])

X_test = pd.read_csv(os.path.join(data_path, 'X_test_final.csv'),
                     header=0, sep=',')
X_test = X_test.drop(columns=['number_sta'])
X_test['baseline_forecast'] = X_test['baseline_forecast'].values + 1

x_train, x_valid, y_train, y_valid = train_test_split(
    X, Y, test_size=0.25, random_state=1)

x_train['baseline_forecast'] = x_train['baseline_forecast'].values + 1
x_valid['baseline_forecast'] = x_valid['baseline_forecast'].values + 1
y_train['Ground_truth'] = y_train['Ground_truth'] + 1
y_valid['Ground_truth'] = y_valid['Ground_truth'] + 1

scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(x_train)
scaled_X_valid = scaler.transform(x_valid)
scaled_X_test = scaler.transform(X_test)

# delete column Id
scaled_X_train = np.delete(scaled_X_train, 1, axis=1)
scaled_X_valid = np.delete(scaled_X_valid, 1, axis=1)
scaled_X_test = np.delete(scaled_X_test, 1, axis=1)

es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)
optimizer = keras.optimizers.RMSprop(learning_rate=0.001)

mlp = keras.models.Sequential()
for _ in range(15):
    mlp.add(keras.layers.Dense(32, activation='relu',
                               kernel_initializer='he_uniform'))
mlp.add(keras.layers.Dense(1, activation='relu'))
mlp.compile(loss="mae", optimizer=optimizer, metrics=["mae"])

history = mlp.fit(scaled_X_train, y_train['Ground_truth'].values, epochs=30, batch_size=32,
                  validation_data=(
                      scaled_X_valid, y_valid['Ground_truth'].values),
                  callbacks=[es])

mlp.save(os.path.join(results_path, 'mlp_model'))

y_pred = mlp.predict(scaled_X_test)
y_pred = np.ravel(y_pred)

Y_pred = pd.DataFrame({'Id': X_test['Id'].values, 'Prediction': y_pred})
Y_pred['Prediction'] = Y_pred.apply(round_values, axis=1)

Y_pred.to_csv(os.path.join(results_path, 'predictions.csv'), index=False)

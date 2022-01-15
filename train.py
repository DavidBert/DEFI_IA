import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from utils import funcs

warnings.simplefilter(action='ignore')


def round_values(row):
    value = row['Prediction']
    return max(value, 1.0)


# Add command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_path', help='Folder to data files')
parser.add_argument(
    '--output_folder', help='Folder to save the model, training logs, and predictions',
)
args = parser.parse_args()

data_path = args.data_path
results_path = args.output_folder

for path in [data_path, results_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# Get Google Drive data
funcs.download_gdrive(data_path)

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

# Scale the values for the neural network
scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(x_train)
scaled_X_valid = scaler.transform(x_valid)
scaled_X_test = scaler.transform(X_test)

# Delete column representing Id
scaled_X_train = np.delete(scaled_X_train, 1, axis=1)
scaled_X_valid = np.delete(scaled_X_valid, 1, axis=1)
scaled_X_test = np.delete(scaled_X_test, 1, axis=1)

es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)
optimizer = keras.optimizers.RMSprop(learning_rate=0.001)

# Create the neural network
mlp = keras.models.Sequential()
for _ in range(15):
    mlp.add(keras.layers.Dense(32, activation='relu',
                               kernel_initializer='he_uniform'))
mlp.add(keras.layers.Dense(1, activation='relu'))
mlp.compile(loss="mae", optimizer=optimizer, metrics=["mae"])

# Fit the model
ts = time.time()
history = mlp.fit(scaled_X_train, y_train['Ground_truth'].values,
                  epochs=50, batch_size=32,
                  validation_data=(
                      scaled_X_valid, y_valid['Ground_truth'].values),
                  callbacks=[es])
te = time.time()

print("Training time: {}".format(te - ts))

funcs.plot_history(history)

# Save the model
mlp.save(os.path.join(results_path, 'mlp_model'))
mlp = keras.models.load_model(os.path.join(results_path, 'mlp_model'))

# Make predictions
y_pred = mlp.predict(scaled_X_test)
y_pred = np.ravel(y_pred)

# Save the predictions as pandas dataframe
Y_pred = pd.DataFrame({'Id': X_test['Id'].values, 'Prediction': y_pred})
Y_pred['Prediction'] = Y_pred.apply(round_values, axis=1)

# Save to CSV file
Y_pred.to_csv(os.path.join(results_path, 'predictions.csv'), index=False)

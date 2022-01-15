import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow import keras
import sys

print("===============================================================================")
print("      start  ....")
data_source = ""
output_folder = ""

data_source_flag = False
output_folder_flag = False
for arg in sys.argv:
    
    if data_source_flag :
        data_source = arg
        data_source_flag = False

    if output_folder_flag:
        output_folder = arg
        output_folder_flag = False 

    if arg == "--data_path":
        data_source_flag = True
    if arg == "--output_folder":
        output_folder_flag = True 


TRAIN_RANGE = 137781
learning_rate = 0.001
batch_size = 256
epochs = 10

print("=====================================================================================")
print("      load trainnig and validation data ")


x_data = pd.read_csv("/home/"+data_source +  "/X_data.csv")
x_data['month'] = pd.DatetimeIndex(x_data['date']).month
x_data = x_data.drop('date',axis=1)
x_data = x_data.drop('number_sta',axis=1)
y_data = pd.read_csv("/home/"+data_source+"/Y_data.csv")


X_data_train = x_data[0:23 * TRAIN_RANGE].copy()
Y_train = y_data[0:TRAIN_RANGE].copy()
X_data_test = x_data[23 * TRAIN_RANGE:]
Y_test = y_data[TRAIN_RANGE:].copy()

scaler = MinMaxScaler()
X_data_train = scaler.fit_transform(X_data_train)
X_data_test  = scaler.fit_transform(X_data_test)


y_scaler = MinMaxScaler()
Y_train = y_scaler.fit_transform(Y_train)
Y_test = y_scaler.fit_transform(Y_test)


dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    X_data_train,
    Y_train,
    sequence_length=23,
    sampling_rate=1,
    batch_size=batch_size,
)
dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    X_data_test,
    Y_test,
    sequence_length=23,
    sampling_rate=1,
    batch_size=batch_size,
)

print("======================================================================================")
print("      create the model ")

for batch in dataset_train.take(1):
    inputs, targets = batch

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(23)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

print("======================================================================================")
print("      Trainning ")

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)


print("=======================================================================================")
print("      load test data ")


x_test_data = pd.read_csv("/home/"+data_source+"/X_test.csv")

_test_data = x_test_data.drop('Unnamed: 0',axis=1)
x_test_data = x_test_data.drop('station',axis=1)
x_test_data = x_test_data.drop('hour',axis=1)
x_test_data = x_test_data.drop('Id',axis=1)
x_test_data = x_test_data.drop('day_index',axis=1)

x_test_data = x_test_data[["ff", "t", "td","hu","dd","precip","month"]]

x_test_data = scaler.fit_transform(x_test_data)

X_test = []
for i in range(0, x_test_data.shape[0]-1, 23):
    tmp = x_test_data[i:i+23]
    X_test.append(tmp)

x_test = np.array(X_test)

print("===================================================================================================")
print("      predict ")


pred = model.predict(x_test)
pred_normal = y_scaler.inverse_transform(pred)
pred_normal
pred_normal = pred_normal.reshape(96012)

tmp_test_data = pd.read_csv("/home/"+data_source+"/X_test.csv")
Ids = []
np_test_data = np.array(tmp_test_data)
for i in range(0, np_test_data.shape[0], 23):
    tmp = np_test_data[i:i+23]
    Ids.append(str(tmp[0,9])+'_'+str(tmp[0,10]))

ids = np.array(Ids)
errors_predictions = pd.DataFrame({'Id': ids, 'prediction': pred_normal})
errors_predictions.to_csv(r'/home/'+output_folder+'/errors_predictions.csv', index=False)

print("=====================================================================================================")
print("=====================================================================================================")
print("\n\n      The predicted errors file: "+output_folder+"/errors_predictions.csv")
print("\n\n=====================================================================================================")
print("=====================================================================================================")
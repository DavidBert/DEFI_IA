import tensorflow
tensorflow.get_logger().setLevel('INFO')
from tensorflow import keras
import keras.backend as K
import numpy as np


def MAPELoss(y_true, y_pred): 
  m = 100 * K.mean(K.abs((y_true - y_pred) / (y_true+1)), axis=-1)
  return m 

def get_model(shape,n_layers,n_neurons): 
    
    # shape : nb de variables explicatives 
    # n_layers : nb de couches
    # n_neurons : neurons/ couche
   
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape))
    
    for i in range(n_layers): 
            model.add(keras.layers.Dense(n_neurons, activation='relu'))
   
    model.add(keras.layers.Dense(1, name='Output'))
    model.compile(optimizer = 'adam', 
                  loss      = 'mae',
                  metrics   = ['mae', 'mse'] )
    return model
    
# Initialisation of the model
def init_model(shape, n_layers = 20, n_neurons = 32): 
    model_reg_nn = get_model(shape, n_layers, n_neurons)
    return model_reg_nn

# Function to evaluate the model
def MAPEVal(y_pred, y_true):
    n = np.shape(y_true)[0]
    m = 100/n*sum(abs((y_true[i] - y_pred[i])/(y_true[i]+1)) for i in range(n))
    return m
    

    

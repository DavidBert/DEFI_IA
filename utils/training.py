import model
import pandas as pd
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
 
def train_model(X, y, model, batch_size, epochs): 
    history = model.fit(X,
                    y,
                    epochs          = epochs,
                    batch_size      = batch_size,
                    verbose         = 0)
    return model, history
    

def plot_history_function(history): 
    # Plot evolution of losses 
    df= pd.DataFrame(data=history.history)
    plot_history(df  ,style = '-', interpolate = True , graphs_per_row = 3)
    plt.show()
 
def main(X, y): 
    shape = len(X.columns)
    model_init = model.init_model(shape)
    batch_size = 128
    epochs = 50
    model_trained, history = train_model(X,y, model_init, batch_size, epochs)
    plot_history_function(history)
    
    return model_trained 
    
    
    

import pandas as pd
import tensorflow
from tensorflow import keras

def prediction(model, X_test, filename): 

    y_pred = model.predict(X_test.drop(["Id"], axis = 1))
    y_pred  = pd.DataFrame(y_pred)
    
    # Creation of the dataframe to submit 
    df_submit = X_test["Id"].to_frame()
    df_submit["Prediction"] = y_pred

    # We put all the negative predictions to 0, and we add 1
    df_submit["Prediction"] = df_submit.Prediction.apply(lambda x : max(0,x) + 1)

    # Export to csv 
    df_submit.to_csv(filename, index = False)
    
    return df_submit
    
    

## Useful functions

import pandas as pd
import numpy as np

def mape(y_true : iter, y_pred: iter) -> float:
    """Compute the mape used by the competition
    Input:
        y_true: real values
        y_pred: predicted values
    Output:
        mape score
    """
    n = len(y_pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    res = abs(y_pred  - (y_true)) / (y_true + 1)
    return (np.sum(res) * 100) / n

def make_submission_csv(ids: pd.Series, pred: pd.Series) -> pd.DataFrame:
    """Save predictions in desired folder
    Input:
        output_path: path to folder where to save the predictions
    """
    sub = pd.DataFrame()
    sub['Id'] = ids
    sub['Predictions'] = pred
    return sub

def save_model(output_path: str, model) -> None:
    params = model.get_params()

def check_path(path: str) -> str:
    if(not path.endswith('/')):
        path = path + '/'
    return path
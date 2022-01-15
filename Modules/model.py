## Class to init, train and make predictions with the model

import pandas as pd
import xgboost as xgb

class Model():
    def __init__(self) -> None:
        """Initialize xgb regressor model"""
        self.model = xgb.XGBRegressor()

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model
        Input:
            X_train: train DataFrame
            y_train: real values
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> iter:
        """Make predictions
        Input:
            X_test: test DataFrame used to make predictions
        """
        pred = self.model.predict(X_test)
        return pred
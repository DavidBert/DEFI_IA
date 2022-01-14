### Class to fill nan and prepare dataset

import pandas as pd

class Preprocess:

    def remove_unused_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns we won't use for the model
        Input:
            df: train DataFrame
        Output:
            DataFrame with only relevant columns
        """
        cols_to_drop = ['ff', 'td', 'hu', 't']
        if 'date' and 'number_sta' in df.columns:
            cols_to_drop += ['date', 'number_sta']
        result = df.drop(cols_to_drop, axis=1)
        
        return result

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace nan with relevant values
        Input:
            df: DataFrame to be cleaned
        Output:
            DataFrame cleaned
        """
        columns = df.columns
        if 'Unnamed: 0' in columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)

        values = {"precip": df['precip'].mean(), "dd": df['dd'].mean()}
        result = df.fillna(value=values)
        
        return result
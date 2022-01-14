## Class to shape dataset and compute features

import pandas as pd

class Feature:

    def concat_train_test(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        """Append test to train to calculate the features ones
        Input:
            X_train: train DataFrame
            X_test: test DataFrame
        Oupput:
            Concatenation of DataFrames with column train in addition to differentiate them
        """
        X_train['train'] = 1
        X_test['train'] = 0
        df = X_train.append(X_test, ignore_index=True, sort=True)
        return df

    def group_by_day(self, data : pd.DataFrame) -> pd.DataFrame:
        """Group the dataframe by day
        Input:
            df : DataFrame to be group
        Output:
            DataFrame grouped
        """
        df = data.copy()
        df['Id'] = df['Id'].apply(lambda x: '_'.join(x.split('_')[0:2]))

        dict_agg = {'dd':'max', 'precip':'sum', 'month':'max', 'train':'max'}

        df = df.groupby('Id').agg(dict_agg).reset_index()
        return df
    
    def compute_rain(self, df: pd.DataFrame) -> pd.Series:
        """Compute 1 if precip > 0 else 0
        Input:
            df : train DataFrame with precip as column
        Output:
            rain feauture as pd.Series
        """
        rain = df.apply(lambda x : 1 if x['precip'] > 0 else 0, axis = 1)
        return rain
    
    def compute_seasons(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract seasons (autumn, spring, summer, winter)
        Input:
            df : train DataFrame with date as columns
        Output:
            seasons as pd.DataFrame
        """
        months = df['month'].apply(lambda x : 'winter' if (x == 1 or x == 2 or x == 12)
                                            else 'spring' if (x == 3 or x == 4 or x == 5) 
                                            else 'summer' if (x == 6 or x == 7 or x == 8) 
                                            else 'autumn')
        result = pd.get_dummies(months)
        result = result[['autumn', 'spring', 'summer', 'winter']]
        return result
    
    def compute_wind_direction(self, df: pd.DataFrame) -> pd.Series:
        """Compute wind direction (0 if degres between 90 and 270)
        Input:
            df : train DataFrame with dd as column
        Output:
            wind direction as pd.Series
        """
        wind_direction = df['dd'].apply(lambda x : 0 if 90 <= x < 270 else 1)
        return wind_direction
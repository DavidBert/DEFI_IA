import pandas as pd
from sklearn.preprocessing import LabelEncoder

def train_imputation(df):
    
    # Version 1 : DropNaNs
    # df = df.dropna()
    
    # Version 2 : KNNImputer
    # from sklearn.impute import KNNImputer
    # temp = df[["Id","number_sta","month"]]
    # imputer = KNNImputer(n_neighbors=2)
    # df = pd.DataFrame(imputer.fit_transform(df[["ff","t","td","hu","dd","precip","lat","lon","height_sta"]]))
    # df = pd.concat([temp,df],axis=1)
    # df.columns = ["Id","number_sta","month","ff","t","td","hu","dd","precip","lat","lon","height_sta"]
    
    # Version 3 : IterativeImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    temp = df[["Id","number_sta","month","Ground_truth"]]
    imp_mean = IterativeImputer(random_state=0)
    df = pd.DataFrame(imp_mean.fit_transform(df[["ff","t","td","hu","dd","precip","lat","lon","height_sta"]]))
    df = pd.concat([temp,df],axis=1)
    df.columns = ["Id","number_sta","month","Ground_truth","ff","t","td","hu","dd","precip","lat","lon","height_sta"]
    
    # Encodage de la seule variable qualitative du dataset
    encoder = LabelEncoder()
    df["number_sta"] = encoder.fit_transform(df["number_sta"].astype(int))
    
    return df
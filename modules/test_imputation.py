import pandas as pd
from sklearn.preprocessing import LabelEncoder

def test_imputation(df):
    
    # Same as train without "Ground_truth"
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    temp = df[["Id","number_sta","month"]]
    imp_mean = IterativeImputer(random_state=0)
    df = pd.DataFrame(imp_mean.fit_transform(df[["ff","t","td","hu","dd","precip","lat","lon","height_sta"]]))
    df = pd.concat([temp,df],axis=1)
    df.columns = ["Id","number_sta","month","ff","t","td","hu","dd","precip","lat","lon","height_sta"]
    
    # Encodage de la seule variable qualitative du dataset
    encoder = LabelEncoder()
    df["number_sta"] = encoder.fit_transform(df["number_sta"].astype(int))
    
    return df
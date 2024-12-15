import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df):
    df.fillna(df.median(), inplace=True)

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

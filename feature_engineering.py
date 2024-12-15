import pandas as pd
import numpy as np


def create_time_features(df, time_column):
    df[time_column] = pd.to_datetime(df[time_column])
    df["year"] = df[time_column].dt.year
    df["month"] = df[time_column].dt.month
    df["day"] = df[time_column].dt.day
    df["weekday"] = df[time_column].dt.weekday
    return df


def add_custom_features(df):
    # Example: ratio or interaction features
    df['balance_to_income_ratio'] = df['balance'] / (df['income'] + 1e-5)
    return df

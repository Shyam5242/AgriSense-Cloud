"""
preprocessing.py
------------------
Implements Algorithm 1 (Agronomic + Satellite Data Preprocessing) from the
project report: missing-value imputation, IQR outlier clipping, and
Min-Max scaling.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def impute_median(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].fillna(df[c].median())
    return df


def clip_outliers_iqr(df: pd.DataFrame, cols, k=1.5) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        df[c] = df[c].clip(lower, upper)
    return df


def scale_train_test(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


def preprocess_yield_table(df: pd.DataFrame, feature_cols):
    df = impute_median(df, feature_cols)
    df = clip_outliers_iqr(df, feature_cols)
    return df

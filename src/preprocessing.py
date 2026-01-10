import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

FEATURES = [
    'Open', 'High', 'Low', 'Close',
    'Volume',
    'Returns', 'ma_10', 'ma_20', 'volatility_10'
]

TARGET = 'Adj Close'

TRAIN_RATIO = 0.8  # 80% train, 20% test (time-based)

SCALER_DIR = Path("../models/scalers")
SCALER_DIR.mkdir(parents=True, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

# TIME-BASED SPLIT
def time_based_split(df: pd.DataFrame):
    split_idx = int(len(df) * TRAIN_RATIO)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    return train_df, test_df

# SCALING
def scale_features(train_df, test_df):
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train_df[FEATURES])

    X_train_scaled = feature_scaler.transform(train_df[FEATURES])
    X_test_scaled = feature_scaler.transform(test_df[FEATURES])

    joblib.dump(feature_scaler, SCALER_DIR / "feature_scaler.joblib")

    return X_train_scaled, X_test_scaled

def scale_target(train_df, test_df):
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_df[[TARGET]])

    y_train_scaled = target_scaler.transform(train_df[[TARGET]])
    y_test_scaled = target_scaler.transform(test_df[[TARGET]])

    joblib.dump(target_scaler, SCALER_DIR / "target_scaler.joblib")

    return y_train_scaled, y_test_scaled

# MAIN PREPROCESS PIPELINE
def preprocess(csv_path: str):
    df = load_data(csv_path)

    train_df, test_df = time_based_split(df)

    X_train, X_test = scale_features(train_df, test_df)
    y_train, y_test = scale_target(train_df, test_df)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_df,
        test_df
    )

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, train_df, test_df = preprocess(
        "../data/processed_data/feat_engg.csv"
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape :", y_test.shape)
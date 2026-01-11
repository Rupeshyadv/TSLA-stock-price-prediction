import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt


WINDOW_SIZE = 60
MODEL_PATH = "models/lstm_tsla.h5"
SCALER_X_PATH = "models/scalers/feature_scaler.joblib"
SCALER_Y_PATH = "models/scalers/target_scaler.joblib"
DATA_PATH = "data/processed_data/feat_engg.csv"

FEATURES = [
    'Open', 'High', 'Low', 'Close',
    'Volume',
    'Returns', 'ma_10', 'ma_20', 'volatility_10'
]

TARGET = 'Adj Close'


# CACHE HEAVY OBJECTS
@st.cache_resource
def load_model_and_scalers():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    feature_scaler = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    return model, feature_scaler, scaler_y

# UTILS
def prepare_last_window(df, feature_scaler):
    """
    Extract last WINDOW_SIZE rows and scale them
    """
    X = df[FEATURES].values[-WINDOW_SIZE:]
    X_scaled = feature_scaler.transform(X)
    return np.expand_dims(X_scaled, axis=0)


def inverse_scale(preds, scaler_y):
    """
    Inverse scale predictions back to price
    """
    preds = preds.reshape(-1, 1)
    return scaler_y.inverse_transform(preds).flatten()


# STREAMLIT UI
st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    layout="centered"
)

st.title("📈 Tesla Stock Price Prediction")
st.markdown("**Predict next 1, 5, and 10 day closing prices using LSTM**")

# Load everything
model, feature_scaler, scaler_y = load_model_and_scalers()

# Load data
df = pd.read_csv(DATA_PATH)

predict_btn = st.button("Predict Next Days")

# PREDICTION
if predict_btn:
    X_input = prepare_last_window(df, feature_scaler)
    preds_scaled = model.predict(X_input)[0]

    preds = inverse_scale(preds_scaled, scaler_y)

    st.subheader("🔮 Predicted Closing Prices")

    col1, col2, col3 = st.columns(3)
    col1.metric("1 Day Ahead", f"${preds[0]:.2f}")
    col2.metric("5 Days Ahead", f"${preds[1]:.2f}")
    col3.metric("10 Days Ahead", f"${preds[2]:.2f}")
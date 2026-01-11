import numpy as np
import joblib
from tensorflow.keras.models import load_model
from pathlib import Path

from preprocessing import preprocess
from sequence_generator import create_sequences

MODEL_PATH = "../models/lstm_tsla.h5"
SCALER_DIR = Path("../models/scalers")

LOOKBACK = 60
HORIZONS = (1, 5, 10)

# LOAD SCALERS
def load_scalers():
    feature_scaler = joblib.load(SCALER_DIR / "feature_scaler.joblib")
    target_scaler = joblib.load(SCALER_DIR / "target_scaler.joblib")
    return feature_scaler, target_scaler

# INVERSE SCALE PREDICTIONS
def inverse_scale_predictions(predictions, target_scaler):
    """
    predictions shape: (samples, num_horizons)
    """
    inversed = []

    for i in range(predictions.shape[1]):
        col = predictions[:, i].reshape(-1, 1)
        inversed.append(target_scaler.inverse_transform(col))

    return np.hstack(inversed)

# EVALUATION PIPELINE
def evaluate(csv_path):
    # Load trained model
    model = load_model(MODEL_PATH, compile=False)

    # Load scalers
    _, target_scaler = load_scalers()

    # Preprocess data
    X_train, X_test, y_train, y_test, train_df, test_df = preprocess(csv_path)

    # Create sequences (ONLY test data)
    X_test_seq, y_test_seq = create_sequences(
        X_test,
        y_test,
        lookback=LOOKBACK,
        horizons=HORIZONS
    )

    # Predict
    y_pred_scaled = model.predict(X_test_seq)

    # Inverse scaling
    y_pred = inverse_scale_predictions(y_pred_scaled, target_scaler)
    y_true = inverse_scale_predictions(y_test_seq, target_scaler)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "horizons": HORIZONS
    }

if __name__ == "__main__":
    results = evaluate("../data/processed_data/feat_engg.csv")

    print("Prediction shape:", results["y_pred"].shape)
    print("Actual shape    :", results["y_true"].shape)

    print("\nSample prediction (1, 5, 10 days):")
    print("Pred:", results["y_pred"][0])
    print("True:", results["y_true"][0])

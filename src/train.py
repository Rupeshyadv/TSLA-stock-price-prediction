import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocessing import preprocess
from sequence_generator import create_sequences
from model import build_lstm_model


CSV_PATH = "../data/processed_data/feat_engg.csv"

LOOKBACK = 60
HORIZONS = (1, 5, 10)

BATCH_SIZE = 32
EPOCHS = 50

MODEL_PATH = "../models/lstm_tsla.h5"

# LOAD & PREPROCESS DATA
X_train, X_test, y_train, y_test, _, _ = preprocess(CSV_PATH)

# CREATE SEQUENCES
X_train_seq, y_train_seq = create_sequences(
    X_train, y_train, lookback=LOOKBACK, horizons=HORIZONS
)

X_test_seq, y_test_seq = create_sequences(
    X_test, y_test, lookback=LOOKBACK, horizons=HORIZONS
)

print("Train sequences:", X_train_seq.shape, y_train_seq.shape)
print("Test sequences :", X_test_seq.shape, y_test_seq.shape)

# BUILD MODEL
model = build_lstm_model(
    input_shape=(LOOKBACK, X_train_seq.shape[2])
)

model.summary()

# CALLBACKS
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True
)

# TRAIN MODEL
history = model.fit(
    X_train_seq,
    y_train_seq,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint],
    verbose=1
)
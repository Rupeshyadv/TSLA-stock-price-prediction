from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# STACKED-LSTM MODEL FOR MULTI-HORIZON STOCK PREDICTION
def build_lstm_model(
    input_shape,
    lstm_units=64,
    dropout_rate=0.2,
    learning_rate=0.001
):
    """
        Parameters
        ----------
        input_shape : tuple
            Shape of input data -> (lookback, num_features)

        lstm_units : int
            Number of LSTM units per layer

        dropout_rate : float
            Dropout for regularization

        learning_rate : float
            Learning rate for Adam optimizer

        Returns
        -------
        model : keras Model
            Compiled LSTM model
    """

    model = Sequential()

    # First LSTM Layer (returns sequences)
    model.add(
        LSTM(
            units=lstm_units,
            return_sequences=True,
            input_shape=input_shape
        )
    )
    model.add(Dropout(dropout_rate))

    # Second LSTM Layer
    model.add(
        LSTM(
            units=lstm_units,
            return_sequences=False
        )
    )
    model.add(Dropout(dropout_rate))

    # Output Layer (Multi-Horizon)
    model.add(Dense(3))  # Predict 1, 5, 10 days

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )

    return model
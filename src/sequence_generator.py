import numpy as np

# CREATE SEQUENCES FOR LSTM (MULTI-HORIZON)
def create_sequences(
    X,
    y,
    lookback=60,
    horizons=(1, 5, 10)
):
    """
        Parameters
        ----------
        X : np.ndarray
            Scaled feature matrix of shape (n_samples, n_features)

        y : np.ndarray
            Scaled target array of shape (n_samples, 1)

        lookback : int
            Number of past time steps used as input

        horizons : tuple
            Future steps to predict (e.g. 1, 5, 10)

        Returns
        -------
        X_seq : np.ndarray
            Shape -> (num_sequences, lookback, n_features)

        y_seq : np.ndarray
            Shape -> (num_sequences, len(horizons))
    """

    X_sequences = []
    y_sequences = []

    max_horizon = max(horizons)

    for i in range(lookback, len(X) - max_horizon):
        # Input sequence (past)
        X_sequences.append(X[i - lookback:i])

        # Multi-horizon targets (future)
        y_sequences.append([
            y[i + h - 1][0] for h in horizons
        ])

    return (
        np.array(X_sequences),
        np.array(y_sequences)
    )
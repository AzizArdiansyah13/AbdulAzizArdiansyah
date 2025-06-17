
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def run_hybrid_forecast(ticker, start_date, end_date, seq_len=60, epochs=5):
    df = yf.download(ticker, start=start_date, end=end_date)[['Close']].dropna()

    if len(df) < seq_len + 1:
        return None, None, None, f"Not enough data to generate sequences (minimum required: {seq_len + 1} rows)."

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    def create_seq(data, seq_len):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i-seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X, y = create_seq(scaled, seq_len)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(50),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

    lstm_train_pred = lstm_model.predict(X_train)
    lstm_test_pred = lstm_model.predict(X_test)

    def make_xgb_input(X_seq, lstm_pred):
        last_input = X_seq[:, -1, 0]
        return np.column_stack((last_input, lstm_pred.flatten()))

    X_train_xgb = make_xgb_input(X_train, lstm_train_pred)
    X_test_xgb = make_xgb_input(X_test, lstm_test_pred)

    xgb_model = XGBRegressor(n_estimators=100)
    xgb_model.fit(X_train_xgb, y_train)

    final_scaled_pred = xgb_model.predict(X_test_xgb)
    final_pred = scaler.inverse_transform(final_scaled_pred.reshape(-1, 1))
    true_val = scaler.inverse_transform(y_test)

    mse = mean_squared_error(true_val, final_pred)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(true_val, label="Actual", color='blue')
    ax.plot(final_pred, label="Hybrid Forecast", color='orange')
    ax.set_title(f"Hybrid Forecast vs Actual (MSE: {mse:.4f})")
    ax.legend()
    fig.tight_layout()

    return fig, mse, df, None

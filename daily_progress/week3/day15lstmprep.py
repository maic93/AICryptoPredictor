"""
Day 15 — LSTM Sequence Data Preparation
Week 3: Deep Learning
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

def load_close(ticker="BTC-USD"):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    return df["Close"].squeeze().dropna().values.reshape(-1, 1)

def make_sequences(data: np.ndarray, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def run():
    print("🧠 Day 15 — LSTM Sequence Preparation")
    print("=" * 50)
    prices = load_close("BTC-USD")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    split = int(len(scaled) * 0.8)
    train_data = scaled[:split]
    test_data  = scaled[split - 60:]

    X_train, y_train = make_sequences(train_data)
    X_test,  y_test  = make_sequences(test_data)

    # Reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.reshape(*X_train.shape, 1)
    X_test  = X_test.reshape(*X_test.shape, 1)

    print(f"  Total prices    : {len(prices)}")
    print(f"  Lookback window : 60 days")
    print(f"  X_train shape   : {X_train.shape}")
    print(f"  y_train shape   : {y_train.shape}")
    print(f"  X_test shape    : {X_test.shape}")
    print(f"  y_test shape    : {y_test.shape}")
    print(f"  Price range     : ${float(prices.min()):,.2f} — ${float(prices.max()):,.2f}")
    print("\n✅ Sequences ready for LSTM training!")

if __name__ == "__main__":
    run()

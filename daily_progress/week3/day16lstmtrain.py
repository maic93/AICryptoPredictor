"""
Day 16 — LSTM-style Sliding Window Model (numpy/sklearn, no TensorFlow)
Week 3: Deep Learning
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

os.makedirs("reports", exist_ok=True)
LOOKBACK = 60

def load_sequences(ticker="BTC-USD"):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    prices = df["Close"].squeeze().dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i - LOOKBACK:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:], scaler, df.index[LOOKBACK:]

def plot(actual, predicted, dates, history):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    axes[0].plot(dates[:len(actual)],    actual,    color="#F7931A", linewidth=1.5, label="Actual")
    axes[0].plot(dates[:len(predicted)], predicted, color="#9B59B6", linewidth=1.5,
                 linestyle="--", label="LSTM-window Predicted")
    axes[0].set_title("Day 16 - LSTM-style Sliding Window: BTC Actual vs Predicted",
                      fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(range(len(history)), history, color="#3498DB", linewidth=1.5)
    axes[1].set_title("Training Loss (MSE over increasing data subsets)", fontweight="bold")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("MSE")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = "reports/day16chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("Day 16 - LSTM-style Sliding Window Training on BTC")
    print("=" * 55)
    print(f"  Lookback : {LOOKBACK} days | Model: Ridge regression on sequences")
    print(f"  Note: Lightweight sklearn implementation (no TensorFlow needed)\n")
    X_train, y_train, X_test, y_test, scaler, dates = load_sequences()
    history = []
    for pct in np.linspace(0.3, 1.0, 20):
        n = int(len(X_train) * pct)
        m = Ridge(alpha=1.0)
        m.fit(X_train[:n], y_train[:n])
        p = m.predict(X_train[:n])
        history.append(float(np.mean((p - y_train[:n])**2)))
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds  = scaler.inverse_transform(model.predict(X_test).reshape(-1,1)).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    print(f"  MAE  : ${mean_absolute_error(actual, preds):,.2f}")
    print(f"  RMSE : ${np.sqrt(mean_squared_error(actual, preds)):,.2f}")
    print(f"  R2   : {r2_score(actual, preds):.4f}")
    plot(actual, preds, dates[int(len(dates)*0.8):], history)
    print("\n✅ Day 16 complete!")

if __name__ == "__main__":
    run()

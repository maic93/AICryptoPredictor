"""
Day 18 — Bidirectional LSTM-style Model (numpy/sklearn, no TensorFlow)
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
        window   = scaled[i - LOOKBACK:i, 0]
        forward  = window
        backward = window[::-1]
        X.append(np.concatenate([forward, backward]))  # BiLSTM: both directions
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:], scaler, df.index[LOOKBACK:]

def plot(actual, predicted, dates):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    axes[0].plot(dates[:len(actual)],    actual,    color="#F7931A", linewidth=1.5, label="Actual")
    axes[0].plot(dates[:len(predicted)], predicted, color="#1ABC9C", linewidth=1.5,
                 linestyle="--", label="BiLSTM-style Predicted")
    axes[0].set_title("Day 18 - BiLSTM-style: BTC Actual vs Predicted", fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    sample = np.linspace(0, 1, LOOKBACK)
    axes[1].fill_between(range(LOOKBACK), sample,        alpha=0.5, color="#3498DB", label="Forward context")
    axes[1].fill_between(range(LOOKBACK), sample[::-1],  alpha=0.5, color="#E74C3C", label="Backward context")
    axes[1].set_title("BiLSTM: Forward + Backward Context Combined", fontweight="bold")
    axes[1].set_xlabel("Timestep"); axes[1].set_ylabel("Context weight")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = "reports/day18chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("Day 18 - Bidirectional LSTM-style Model")
    print("=" * 50)
    print("  BiLSTM reads sequences both forward AND backward.")
    print("  Feature size doubles: forward window + reversed window.\n")
    X_train, y_train, X_test, y_test, scaler, dates = load_sequences()
    model = Ridge(alpha=0.5)
    model.fit(X_train, y_train)
    preds  = scaler.inverse_transform(model.predict(X_test).reshape(-1,1)).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    print(f"  Feature size : {X_train.shape[1]} (forward {LOOKBACK} + backward {LOOKBACK})")
    print(f"  MAE  : ${mean_absolute_error(actual, preds):,.2f}")
    print(f"  RMSE : ${np.sqrt(mean_squared_error(actual, preds)):,.2f}")
    print(f"  R2   : {r2_score(actual, preds):.4f}")
    plot(actual, preds, dates[int(len(dates)*0.8):])
    print("\n✅ Day 18 complete!")

if __name__ == "__main__":
    run()

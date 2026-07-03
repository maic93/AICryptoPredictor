"""
Day 15 — LSTM Sequence Data Preparation
Week 3: Deep Learning
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

os.makedirs("reports", exist_ok=True)

def load_close(ticker="BTC-USD"):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    return df["Close"].squeeze().dropna()

def make_sequences(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def plot(prices, close_series):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    axes[0].plot(close_series.index, close_series.values, color="#F7931A", linewidth=1.5)
    axes[0].axvline(close_series.index[int(len(close_series)*0.8)],
                     color="red", linestyle="--", linewidth=1.5, label="Train/Test Split (80/20)")
    axes[0].set_title("BTC Price — Train vs Test Split for LSTM", fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    lookback = 60
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.values.reshape(-1,1))
    X, y = make_sequences(scaled, lookback)
    sample_idx = 0
    axes[1].plot(range(lookback), X[sample_idx], color="#3498DB", linewidth=1.5, label="Input sequence (60 days)")
    axes[1].axvline(lookback-1, color="red", linestyle="--")
    axes[1].scatter([lookback], [y[sample_idx]], color="#E74C3C", s=100, zorder=5, label=f"Target (day 61)")
    axes[1].set_title("Example LSTM Input Sequence to Target", fontweight="bold")
    axes[1].set_xlabel("Timestep"); axes[1].set_ylabel("Scaled Price")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = "reports/day15chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("🧠 Day 15 — LSTM Sequence Preparation")
    print("=" * 50)
    close = load_close("BTC-USD")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close.values.reshape(-1,1))
    split = int(len(scaled) * 0.8)
    X_train, y_train = make_sequences(scaled[:split])
    X_test,  y_test  = make_sequences(scaled[split-60:])
    X_train = X_train.reshape(*X_train.shape, 1)
    X_test  = X_test.reshape(*X_test.shape, 1)
    print(f"  Total prices    : {len(close)}")
    print(f"  Lookback window : 60 days")
    print(f"  X_train shape   : {X_train.shape}")
    print(f"  X_test shape    : {X_test.shape}")
    print(f"  Price range     : ${float(close.min()):,.2f} — ${float(close.max()):,.2f}")
    plot(close, close)
    print("\n✅ Sequences ready for LSTM!")

if __name__ == "__main__":
    run()

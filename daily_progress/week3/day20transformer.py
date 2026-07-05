"""
Day 20 — Transformer-style Self-Attention Model (numpy/sklearn, no TensorFlow)
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

def self_attention(X):
    """Simplified self-attention: softmax(QK^T / sqrt(d)) * V where Q=K=V=X"""
    d = X.shape[1]
    scores = X @ X.T / np.sqrt(d)
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ X

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

def plot(actual, predicted, dates):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    axes[0].plot(dates[:len(actual)],    actual,    color="#F7931A", linewidth=1.5, label="Actual")
    axes[0].plot(dates[:len(predicted)], predicted, color="#8E44AD", linewidth=1.5,
                 linestyle="--", label="Transformer-style Predicted")
    axes[0].set_title("Day 20 - Transformer-style: BTC Actual vs Predicted", fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    # Sample self-attention heatmap
    np.random.seed(42)
    sample = np.random.rand(10, 10)
    scores = sample @ sample.T / np.sqrt(10)
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores); weights /= weights.sum(axis=1, keepdims=True)
    im = axes[1].imshow(weights, cmap="Blues")
    axes[1].set_title("Self-Attention Pattern (10x10 sample)", fontweight="bold")
    axes[1].set_xlabel("Key position"); axes[1].set_ylabel("Query position")
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    out = "reports/day20chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("Day 20 - Transformer-style Self-Attention Model")
    print("=" * 50)
    print("  Transformers use self-attention: every position attends to every other.")
    print("  Implemented here with numpy — same math, no GPU needed.\n")
    X_train, y_train, X_test, y_test, scaler, dates = load_sequences()
    print("  Applying self-attention transform...")
    X_train_attn = self_attention(X_train)
    X_test_attn  = self_attention(X_test)
    model = Ridge(alpha=0.5)
    model.fit(X_train_attn, y_train)
    preds  = scaler.inverse_transform(model.predict(X_test_attn).reshape(-1,1)).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    print(f"  MAE  : ${mean_absolute_error(actual, preds):,.2f}")
    print(f"  RMSE : ${np.sqrt(mean_squared_error(actual, preds)):,.2f}")
    print(f"  R2   : {r2_score(actual, preds):.4f}")
    plot(actual, preds, dates[int(len(dates)*0.8):])
    print("\n✅ Day 20 complete!")

if __name__ == "__main__":
    run()

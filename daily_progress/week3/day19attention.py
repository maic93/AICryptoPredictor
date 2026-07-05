"""
Day 19 — Attention Mechanism (numpy/sklearn, no TensorFlow)
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

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

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

def plot(actual, predicted, attn_weights, dates):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    axes[0].plot(dates[:len(actual)],    actual,    color="#F7931A", linewidth=1.5, label="Actual")
    axes[0].plot(dates[:len(predicted)], predicted, color="#E67E22", linewidth=1.5,
                 linestyle="--", label="Attention-weighted Predicted")
    axes[0].set_title("Day 19 - Attention Mechanism: BTC Actual vs Predicted", fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    bar_colors = plt.cm.YlOrRd(attn_weights / attn_weights.max())
    axes[1].bar(range(LOOKBACK), attn_weights, color=bar_colors)
    axes[1].set_title("Attention Weights per Timestep (higher = more important)", fontweight="bold")
    axes[1].set_xlabel("Timestep (0=oldest, 59=latest)")
    axes[1].set_ylabel("Attention Weight"); axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = "reports/day19chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("Day 19 - Attention Mechanism on Sequence Model")
    print("=" * 50)
    print("  Attention learns WHICH timesteps matter most.\n")
    X_train, y_train, X_test, y_test, scaler, dates = load_sequences()
    # Learn attention weights via correlation with target
    correlations = np.array([np.corrcoef(X_train[:, t], y_train)[0, 1]
                              for t in range(X_train.shape[1])])
    attn_weights = softmax(np.abs(correlations))
    X_train_attn = X_train * attn_weights
    X_test_attn  = X_test  * attn_weights
    model = Ridge(alpha=0.5)
    model.fit(X_train_attn, y_train)
    preds  = scaler.inverse_transform(model.predict(X_test_attn).reshape(-1,1)).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    print(f"  MAE  : ${mean_absolute_error(actual, preds):,.2f}")
    print(f"  RMSE : ${np.sqrt(mean_squared_error(actual, preds)):,.2f}")
    print(f"  R2   : {r2_score(actual, preds):.4f}")
    top3 = np.argsort(attn_weights)[-3:][::-1]
    print(f"  Top 3 attended timesteps: {top3} (0=oldest, 59=latest)")
    plot(actual, preds, attn_weights, dates[int(len(dates)*0.8):])
    print("\n✅ Day 19 complete!")

if __name__ == "__main__":
    run()

"""
Day 25 — Live Price Fetching + Real-Time Prediction Pipeline
Week 4: Production
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from datetime import datetime
import os

COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
COLORS = ["#F7931A", "#627EEA", "#9945FF", "#F0B90B"]
os.makedirs("reports", exist_ok=True)

def build_and_predict(ticker):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True).dropna()
    close = df["Close"].squeeze()
    for lag in [1, 2, 3, 5, 7, 14]:
        df[f"lag{lag}"] = close.shift(lag)
    for w in [7, 14, 30]:
        df[f"rollMean{w}"] = close.rolling(w).mean()
        df[f"rollStd{w}"]  = close.rolling(w).std()
    df["return1d"] = close.pct_change()
    df["return7d"] = close.pct_change(7)
    df["target"] = close.shift(-1)
    df.dropna(inplace=True)
    feat_cols = [c for c in df.columns if c not in ["target","Open","High","Low","Close","Volume"]]
    X, y = df[feat_cols].values, df["target"].values
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0)
    model.fit(X, y)
    latest_price = float(close.iloc[-1].item())
    predicted    = float(model.predict(X[-1:])[0])
    history      = close.iloc[-30:]
    return latest_price, predicted, history

def plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for i, (coin, color) in enumerate(zip(COINS, COLORS)):
        current, predicted, history = results[coin]
        change_pct = (predicted - current) / current * 100
        ax = axes[i]
        ax.plot(history.index, history.values, color=color, linewidth=2, label="Last 30 days")
        ax.scatter([history.index[-1]], [current],   color="white",  s=80, zorder=5)
        ax.scatter([history.index[-1]], [predicted], color="#E74C3C", s=100, zorder=5,
                   label=f"Prediction: ${predicted:,.0f} ({change_pct:+.2f}%)",
                   marker="^" if predicted > current else "v")
        ax.set_title(f"{coin}", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    plt.suptitle(f"Day 25 — Live Predictions | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day25chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("⚡ Day 25 — Live Prediction Pipeline")
    print("=" * 60)
    print(f"  🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    print(f"  {'Coin':<12} {'Current':>14} {'Predicted':>14} {'Signal':>10}")
    print("  " + "-" * 55)
    results = {}
    for coin in COINS:
        current, predicted, history = build_and_predict(coin)
        results[coin] = (current, predicted, history)
        change_pct = (predicted - current) / current * 100
        signal = "📈 BUY" if change_pct > 0.5 else "📉 SELL" if change_pct < -0.5 else "⏸️  HOLD"
        print(f"  {coin:<12} ${current:>12,.2f}  ${predicted:>12,.2f}  {signal} ({change_pct:+.2f}%)")
    plot(results)
    print("\n  ⚠️  Educational only — not financial advice!")
    print("\n✅ Day 25 complete!")

if __name__ == "__main__":
    run()

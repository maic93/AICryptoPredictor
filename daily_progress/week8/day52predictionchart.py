"""
Day 52 — Prediction Chart Component
Week 8: Web Dashboard
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import os

os.makedirs("reports", exist_ok=True)
COINS  = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"]
COLORS = ["#F7931A","#627EEA","#9945FF","#F0B90B"]

def predict(ticker: str):
    df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.dropna(inplace=True)
    close = df["Close"].squeeze()
    for lag in [1,2,3,5,7,14]:
        df[f"lag{lag}"] = close.shift(lag)
    for w in [7,14,30]:
        df[f"rollMean{w}"] = close.rolling(w).mean()
        df[f"rollStd{w}"]  = close.rolling(w).std()
    df["return1d"] = close.pct_change()
    df["target"]   = close.shift(-1)
    df.dropna(inplace=True)
    feat_cols = [str(c) for c in df.columns
                 if c not in ["target","Open","High","Low","Close","Volume"]]
    df.columns = [str(c) for c in df.columns]
    X = df[feat_cols]; y = df["target"]
    split = int(len(X) * 0.8)
    model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X.iloc[:split], y.iloc[:split])
    preds  = model.predict(X.iloc[split:])
    actual = y.iloc[split:].values
    dates  = df.index[split:]
    n      = min(len(preds), len(actual), len(dates))
    mae    = mean_absolute_error(actual[:n], preds[:n])
    next_pred = float(model.predict(X.iloc[[-1]])[0])
    return dates[:n], actual[:n], preds[:n], mae, next_pred, float(close.iloc[-1].item())

def plot(results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor("#0e1117")
    axes = axes.flatten()

    for i, (coin, (dates, actual, preds, mae, next_pred, curr)) in enumerate(results.items()):
        ax = axes[i]
        ax.set_facecolor("#262730")
        label = coin.replace("-USD","")
        color = COLORS[i]
        change = (next_pred - curr) / curr * 100
        ax.plot(dates, actual, color="white",  linewidth=1.2, alpha=0.7, label="Actual")
        ax.plot(dates, preds,  color=color,    linewidth=1.5, linestyle="--", label="Predicted")
        ax.axhline(next_pred, color="#2ECC71" if change > 0 else "#E74C3C",
                    linestyle=":", linewidth=1.5,
                    label=f"Tomorrow: ${next_pred:,.0f} ({change:+.1f}%)")
        ax.set_title(f"{label} | MAE: ${mae:,.0f}", color="white", fontweight="bold")
        ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")
        ax.tick_params(colors="#888", labelsize=7)
        ax.grid(alpha=0.15)
        for sp in ax.spines.values():
            sp.set_color("#444")

    plt.suptitle("Day 52 - Prediction Chart Component (All 4 Coins)",
                  fontsize=13, fontweight="bold", color="white")
    plt.tight_layout()
    out = "reports/day52chart.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 52 - Prediction Chart Component")
    print("=" * 55)
    results = {}
    for coin in COINS:
        try:
            dates, actual, preds, mae, next_pred, curr = predict(coin)
            results[coin] = (dates, actual, preds, mae, next_pred, curr)
            change = (next_pred - curr) / curr * 100
            print(f"  {coin:<12} MAE=${mae:,.0f} | "
                  f"Tomorrow: ${next_pred:,.0f} ({change:+.1f}%)")
        except Exception as e:
            print(f"  {coin:<12} ERROR: {e}")
    plot(results)
    print("\n  Not financial advice!")
    print("\n✅ Day 52 complete!")

if __name__ == "__main__":
    run()

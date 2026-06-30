"""
Day 06 — Feature Engineering: Lag Features & Rolling Stats
Week 1: Foundations
"""
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("reports", exist_ok=True)

def load(ticker="BTC-USD"):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df.dropna()

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()
    for lag in [1, 2, 3, 5, 7, 14, 21]:
        df[f"lag{lag}"] = close.shift(lag)
    for window in [7, 14, 30]:
        df[f"rollMean{window}"] = close.rolling(window).mean()
        df[f"rollStd{window}"]  = close.rolling(window).std()
        df[f"rollMin{window}"]  = close.rolling(window).min()
        df[f"rollMax{window}"]  = close.rolling(window).max()
    df["return1d"]  = close.pct_change(1)
    df["return7d"]  = close.pct_change(7)
    df["return30d"] = close.pct_change(30)
    df["target"] = close.shift(-1)
    df.dropna(inplace=True)
    return df

def plot(df: pd.DataFrame):
    close = df["Close"].squeeze()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    # Rolling means
    axes[0].plot(df.index, close, color="#F7931A", linewidth=1, alpha=0.7, label="Close")
    axes[0].plot(df.index, df["rollMean7"].squeeze(),  color="#3498DB", linewidth=1.5, label="7d MA")
    axes[0].plot(df.index, df["rollMean14"].squeeze(), color="#E74C3C", linewidth=1.5, label="14d MA")
    axes[0].plot(df.index, df["rollMean30"].squeeze(), color="#2ECC71", linewidth=1.5, label="30d MA")
    axes[0].set_title("BTC Price with Rolling Means", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Correlation heatmap of features
    feat_cols = ["lag1", "lag7", "lag14", "rollMean7", "rollMean14", "rollMean30",
                 "rollStd7", "return1d", "return7d", "target"]
    corr = df[feat_cols].corr()
    im = axes[1].imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(feat_cols)))
    axes[1].set_yticks(range(len(feat_cols)))
    axes[1].set_xticklabels(feat_cols, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticklabels(feat_cols, fontsize=8)
    axes[1].set_title("Feature Correlation Heatmap", fontweight="bold")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    out = "reports/day06chart.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("⚙️  Day 06 — Feature Engineering")
    print("=" * 50)
    df = load("BTC-USD")
    df = engineer_features(df)
    feat_cols = [c for c in df.columns if c != "target"]
    print(f"  Total features  : {len(feat_cols)}")
    print(f"  Dataset shape   : {df.shape}")
    print(f"  Sample target   : ${float(df['target'].iloc[-1].item()):,.2f}")
    plot(df)
    print("\n✅ Feature engineering complete!")

if __name__ == "__main__":
    run()

"""
Day 07 — Week 1 Recap & Full Data Pipeline
Week 1: Foundations
"""
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
COLORS = ["#F7931A", "#627EEA", "#9945FF", "#F0B90B"]
os.makedirs("reports", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

def full_pipeline(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True).dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    close = df["Close"].squeeze()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    for lag in [1, 3, 7, 14]:
        df[f"lag{lag}"] = close.shift(lag)
    for w in [7, 14, 30]:
        df[f"rollMean{w}"] = close.rolling(w).mean()
        df[f"rollStd{w}"]  = close.rolling(w).std()
    df["return1d"] = close.pct_change(1)
    df["return7d"]  = close.pct_change(7)
    df["target"] = close.shift(-1)
    df.dropna(inplace=True)
    df.to_csv(f"data/processed/{ticker.replace('-','')}.csv")
    return df

def plot(all_data: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Normalized price comparison
    ax = axes[0, 0]
    for (coin, df), color in zip(all_data.items(), COLORS):
        close = df["Close"].squeeze()
        normalized = close / float(close.iloc[0].item()) * 100
        ax.plot(df.index, normalized, color=color, linewidth=1.5, label=coin)
    ax.set_title("Normalized Price (Base=100)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row count per coin
    ax = axes[0, 1]
    coins = list(all_data.keys())
    rows  = [len(df) for df in all_data.values()]
    bars  = ax.bar(coins, rows, color=COLORS)
    ax.set_title("Dataset Size per Coin", fontweight="bold")
    ax.set_ylabel("Rows")
    for bar, row in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(row), ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # BTC RSI
    ax = axes[1, 0]
    btc = all_data["BTC-USD"]
    ax.plot(btc.index, btc["RSI"].squeeze(), color="#9B59B6", linewidth=1)
    ax.axhline(70, color="red",   linestyle="--", linewidth=0.8)
    ax.axhline(30, color="green", linestyle="--", linewidth=0.8)
    ax.set_title("BTC RSI (14)", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)

    # BTC 30d rolling volatility
    ax = axes[1, 1]
    for (coin, df), color in zip(all_data.items(), COLORS):
        vol = df["Close"].squeeze().pct_change().rolling(30).std() * 100
        ax.plot(df.index, vol, color=color, linewidth=1, label=coin)
    ax.set_title("30-Day Rolling Volatility (%)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle("Day 07 — Week 1 Pipeline Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = "reports/day07chart.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("📝 Day 07 — Week 1 Recap & Pipeline")
    print("=" * 50)
    all_data = {}
    for coin in COINS:
        df = full_pipeline(coin)
        all_data[coin] = df
        print(f"  ✅ {coin:10s} | {df.shape[0]} rows | {df.shape[1]} features")
    plot(all_data)
    print("\n🎉 Week 1 complete! Full pipeline ready.")

if __name__ == "__main__":
    run()

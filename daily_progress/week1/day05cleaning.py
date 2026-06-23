"""
Day 05 — Data Cleaning & Missing Value Handling
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
    return yf.download(ticker, period="2y", progress=False, auto_adjust=True)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"  Before cleaning : {df.shape[0]} rows, {df.isnull().sum().sum()} nulls")
    df = df.ffill()
    df = df.dropna()
    df = df[~df.index.duplicated(keep="first")]
    vol = df["Volume"].squeeze()
    df = df[vol > vol.mean() - 5 * vol.std()]
    for col in ["Open", "High", "Low", "Close"]:
        df = df[df[col].squeeze() > 0]
    print(f"  After cleaning  : {df.shape[0]} rows, {df.isnull().sum().sum()} nulls")
    return df

def plot(coins_data: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    colors = ["#F7931A", "#627EEA", "#9945FF", "#F0B90B"]
    for i, (coin, df) in enumerate(coins_data.items()):
        close = df["Close"].squeeze()
        ax = axes[i]
        ax.plot(df.index, close, color=colors[i], linewidth=1.2)
        ax.set_title(f"{coin} — {len(df)} clean rows", fontweight="bold")
        ax.set_ylabel("Price (USD)")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    plt.suptitle("Day 05 — Cleaned Price Data (2Y)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = "reports/day05chart.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("🧹 Day 05 — Data Cleaning")
    print("=" * 50)
    coins_data = {}
    for coin in ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]:
        print(f"\n  [{coin}]")
        df = load(coin)
        df = clean(df)
        coins_data[coin] = df
    plot(coins_data)
    print("\n✅ All data cleaned!")

if __name__ == "__main__":
    run()

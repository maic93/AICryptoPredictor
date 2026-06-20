"""
Day 03 — Exploratory Data Analysis (EDA)
Week 1: Foundations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def load(ticker="BTC-USD"):
    return yf.download(ticker, period="2y", progress=False, auto_adjust=True)

def eda(df: pd.DataFrame, ticker="BTC-USD"):
    close = df["Close"].squeeze()
    print(f"\n📊 EDA — {ticker}")
    print(f"  Rows        : {len(df)}")
    print(f"  Date range  : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Mean close  : ${float(close.mean()):>12,.2f}")
    print(f"  Max close   : ${float(close.max()):>12,.2f}")
    print(f"  Min close   : ${float(close.min()):>12,.2f}")
    print(f"  Volatility  : {float(close.pct_change().std() * 100):.2f}% daily std")
    print(f"  Null values : {df.isnull().sum().sum()}")

def plot(df: pd.DataFrame, ticker="BTC-USD"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    close = df["Close"].squeeze()
    vol = df["Volume"].squeeze()
    axes[0].plot(df.index, close, color="#F7931A", linewidth=1.5)
    axes[0].set_title(f"{ticker} Closing Price (2Y)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Price (USD)")
    axes[0].grid(alpha=0.3)
    axes[1].bar(df.index, vol, color="#627EEA", alpha=0.6, width=1)
    axes[1].set_title("Volume", fontsize=12)
    axes[1].set_ylabel("Volume")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/btc_eda.png", dpi=150)
    print("  📈 Chart saved → data/btc_eda.png")
    plt.close()

def run():
    print("🔍 Day 03 — Exploratory Data Analysis")
    print("=" * 50)
    for coin in ["BTC-USD", "ETH-USD"]:
        df = load(coin)
        eda(df, coin)
    df_btc = load("BTC-USD")
    plot(df_btc)

if __name__ == "__main__":
    run()

"""
Day 04 — Technical Indicators: RSI, MACD, Bollinger Bands
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

def add_rsi(df: pd.DataFrame, period=14) -> pd.DataFrame:
    close = df["Close"].squeeze()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACDSignal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACDHist"] = df["MACD"] - df["MACDSignal"]
    return df

def add_bollinger(df: pd.DataFrame, period=20) -> pd.DataFrame:
    close = df["Close"].squeeze()
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    df["BBUpper"] = sma + 2 * std
    df["BBMiddle"] = sma
    df["BBLower"] = sma - 2 * std
    return df

def plot(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    close = df["Close"].squeeze()

    # Price + Bollinger Bands
    axes[0].plot(df.index, close, color="#F7931A", linewidth=1.5, label="Close")
    axes[0].plot(df.index, df["BBUpper"].squeeze(), color="#aaa", linewidth=0.8, linestyle="--", label="BB Upper")
    axes[0].plot(df.index, df["BBLower"].squeeze(), color="#aaa", linewidth=0.8, linestyle="--", label="BB Lower")
    axes[0].fill_between(df.index, df["BBUpper"].squeeze(), df["BBLower"].squeeze(), alpha=0.1, color="gray")
    axes[0].set_title("BTC Price + Bollinger Bands", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # RSI
    axes[1].plot(df.index, df["RSI"].squeeze(), color="#9B59B6", linewidth=1.2)
    axes[1].axhline(70, color="red", linestyle="--", linewidth=0.8, label="Overbought (70)")
    axes[1].axhline(30, color="green", linestyle="--", linewidth=0.8, label="Oversold (30)")
    axes[1].set_title("RSI (14)", fontweight="bold")
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # MACD
    axes[2].plot(df.index, df["MACD"].squeeze(), color="#3498DB", linewidth=1.2, label="MACD")
    axes[2].plot(df.index, df["MACDSignal"].squeeze(), color="#E74C3C", linewidth=1.2, label="Signal")
    axes[2].bar(df.index, df["MACDHist"].squeeze(), color="#2ECC71", alpha=0.4, width=1, label="Histogram")
    axes[2].set_title("MACD", fontweight="bold")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = "reports/day04chart.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("📊 Day 04 — Technical Indicators")
    print("=" * 50)
    df = load("BTC-USD")
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df.dropna(inplace=True)

    latest = df.iloc[-1]
    print(f"  RSI (14)        : {float(latest['RSI'].item()):.2f}")
    print(f"  MACD            : {float(latest['MACD'].item()):.2f}")
    print(f"  MACD Signal     : {float(latest['MACDSignal'].item()):.2f}")
    print(f"  BB Upper        : ${float(latest['BBUpper'].item()):,.2f}")
    print(f"  BB Lower        : ${float(latest['BBLower'].item()):,.2f}")
    print(f"\n✅ Indicators added. Shape: {df.shape}")
    plot(df)

if __name__ == "__main__":
    run()

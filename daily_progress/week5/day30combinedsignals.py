"""
Day 30 — RSI + MACD Combined Signal Strategy
Week 5: Live Trading Signals
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("reports", exist_ok=True)

def load(ticker="BTC-USD"):
    df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df.dropna()

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()
    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]       = ema12 - ema26
    df["MACDSignal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    # Combined signal
    rsi_buy   = df["RSI"] < 40
    rsi_sell  = df["RSI"] > 60
    macd_buy  = df["MACD"] > df["MACDSignal"]
    macd_sell = df["MACD"] < df["MACDSignal"]
    df["Signal"] = 0
    df.loc[rsi_buy  & macd_buy,  "Signal"] =  1   # BUY
    df.loc[rsi_sell & macd_sell, "Signal"] = -1   # SELL
    df.dropna(inplace=True)
    return df

def plot(df: pd.DataFrame):
    close  = df["Close"].squeeze()
    buys   = df[df["Signal"] ==  1]
    sells  = df[df["Signal"] == -1]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(df.index, close, color="#F7931A", linewidth=1.2, label="BTC Price")
    axes[0].scatter(buys.index,  buys["Close"].squeeze(),  marker="^", color="#2ECC71",
                    s=80, zorder=5, label="BUY signal")
    axes[0].scatter(sells.index, sells["Close"].squeeze(), marker="v", color="#E74C3C",
                    s=80, zorder=5, label="SELL signal")
    axes[0].set_title("BTC Price with Combined RSI+MACD Signals", fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(df.index, df["RSI"].squeeze(), color="#9B59B6", linewidth=1.2)
    axes[1].axhline(60, color="#E74C3C", linestyle="--", linewidth=0.8)
    axes[1].axhline(40, color="#2ECC71", linestyle="--", linewidth=0.8)
    axes[1].set_title("RSI (14)", fontweight="bold")
    axes[1].set_ylim(0, 100); axes[1].grid(alpha=0.3)

    axes[2].plot(df.index, df["MACD"].squeeze(),       color="#3498DB", linewidth=1.2, label="MACD")
    axes[2].plot(df.index, df["MACDSignal"].squeeze(), color="#E74C3C", linewidth=1.2, label="Signal")
    axes[2].bar(df.index, (df["MACD"] - df["MACDSignal"]).squeeze(),
                color="#2ECC71", alpha=0.4, width=1)
    axes[2].set_title("MACD", fontweight="bold")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = "reports/day30chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 30 - RSI + MACD Combined Signal Strategy")
    print("=" * 55)
    df = load("BTC-USD")
    df = compute_signals(df)
    buys  = (df["Signal"] ==  1).sum()
    sells = (df["Signal"] == -1).sum()
    print(f"  Period      : {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  BUY signals : {buys}")
    print(f"  SELL signals: {sells}")
    print(f"  Latest RSI  : {float(df['RSI'].iloc[-1].item()):.2f}")
    print(f"  Latest MACD : {float(df['MACD'].iloc[-1].item()):.4f}")
    sig = df["Signal"].iloc[-1]
    print(f"  Today signal: {'BUY' if sig == 1 else 'SELL' if sig == -1 else 'HOLD'}")
    plot(df)
    print("\n✅ Day 30 complete!")

if __name__ == "__main__":
    run()

"""
Day 07 — Week 1 Recap & Full Data Pipeline
Week 1: Foundations
"""
import pandas as pd
import numpy as np
import yfinance as yf
import os

COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
os.makedirs("data/processed", exist_ok=True)

def full_pipeline(ticker: str) -> pd.DataFrame:
    # Fetch
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    df.dropna(inplace=True)

    close = df["Close"].squeeze()

    # Indicators
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    # Features
    for lag in [1, 3, 7, 14]:
        df[f"lag{lag}"] = close.shift(lag)
    for w in [7, 14, 30]:
        df[f"rollMean{w}"] = close.rolling(w).mean()
        df[f"rollStd{w}"]  = close.rolling(w).std()
    df["return1d"]  = close.pct_change(1)
    df["return7d"]  = close.pct_change(7)

    # Target
    df["target"] = close.shift(-1)
    df.dropna(inplace=True)

    out = f"data/processed/{ticker.replace('-','')}.csv"
    df.to_csv(out)
    return df

def run():
    print("📝 Day 07 — Week 1 Recap & Pipeline")
    print("=" * 50)
    for coin in COINS:
        df = full_pipeline(coin)
        print(f"  ✅ {coin:10s} | {df.shape[0]} rows | {df.shape[1]} features | saved")
    print("\n🎉 Week 1 complete! Full pipeline ready.")
    print("   Next: Week 2 — Classical Machine Learning")

if __name__ == "__main__":
    run()

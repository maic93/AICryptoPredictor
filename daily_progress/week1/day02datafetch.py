"""
Day 02 — Fetch Historical Crypto Data
Week 1: Foundations
"""
import yfinance as yf
import pandas as pd
import os

COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
PERIOD = "2y"
os.makedirs("data/raw", exist_ok=True)

def fetch(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=PERIOD, progress=False, auto_adjust=True)
    df.dropna(inplace=True)
    df.to_csv(f"data/raw/{ticker.replace('-','')}.csv")
    return df

def run():
    print("📥 Day 02 — Fetching Historical Crypto Data")
    print("=" * 50)
    for coin in COINS:
        df = fetch(coin)
        latest = float(df["Close"].iloc[-1])
        print(f"  ✅ {coin:10s} | {len(df):4d} rows | Latest close: ${latest:,.2f}")
    print("\n💾 Data saved to data/raw/")

if __name__ == "__main__":
    run()

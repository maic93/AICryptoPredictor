"""
Day 05 — Data Cleaning & Missing Value Handling
Week 1: Foundations
"""
import pandas as pd
import numpy as np
import yfinance as yf

def load(ticker="BTC-USD"):
    return yf.download(ticker, period="2y", progress=False, auto_adjust=True)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"  Before cleaning : {df.shape[0]} rows, {df.isnull().sum().sum()} nulls")
    # Forward fill missing prices (weekend gaps in some feeds)
    df = df.ffill()
    # Drop any remaining nulls
    df = df.dropna()
    # Remove duplicate indices
    df = df[~df.index.duplicated(keep="first")]
    # Remove extreme outliers in volume (> 5 std)
    vol = df["Volume"].squeeze()
    vol_mean, vol_std = vol.mean(), vol.std()
    df = df[vol > vol_mean - 5 * vol_std]
    # Ensure positive prices
    for col in ["Open", "High", "Low", "Close"]:
        df = df[df[col].squeeze() > 0]
    print(f"  After cleaning  : {df.shape[0]} rows, {df.isnull().sum().sum()} nulls")
    return df

def run():
    print("🧹 Day 05 — Data Cleaning")
    print("=" * 50)
    for coin in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        print(f"\n  [{coin}]")
        df = load(coin)
        df = clean(df)
    print("\n✅ All data cleaned!")

if __name__ == "__main__":
    run()

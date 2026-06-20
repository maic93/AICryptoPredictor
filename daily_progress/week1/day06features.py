"""
Day 06 — Feature Engineering: Lag Features & Rolling Stats
Week 1: Foundations
"""
import pandas as pd
import numpy as np
import yfinance as yf

def load(ticker="BTC-USD"):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    return df.dropna()

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].squeeze()

    # Lag features (past N days prices)
    for lag in [1, 2, 3, 5, 7, 14, 21]:
        df[f"lag{lag}"] = close.shift(lag)

    # Rolling statistics
    for window in [7, 14, 30]:
        df[f"rollMean{window}"] = close.rolling(window).mean()
        df[f"rollStd{window}"]  = close.rolling(window).std()
        df[f"rollMin{window}"]  = close.rolling(window).min()
        df[f"rollMax{window}"]  = close.rolling(window).max()

    # Price momentum
    df["return1d"]  = close.pct_change(1)
    df["return7d"]  = close.pct_change(7)
    df["return30d"] = close.pct_change(30)

    # Target: next day's closing price
    df["target"] = close.shift(-1)

    df.dropna(inplace=True)
    return df

def run():
    print("⚙️  Day 06 — Feature Engineering")
    print("=" * 50)
    df = load("BTC-USD")
    df = engineer_features(df)
    feature_cols = [c for c in df.columns if c != "target"]
    print(f"  Total features  : {len(feature_cols)}")
    print(f"  Dataset shape   : {df.shape}")
    print(f"  Feature list    : {feature_cols[:8]} ...")
    print(f"  Sample target   : ${float(df['target'].iloc[-1]):,.2f}")
    print("\n✅ Feature engineering complete!")

if __name__ == "__main__":
    run()

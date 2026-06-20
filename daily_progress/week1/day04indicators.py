"""
Day 04 — Technical Indicators: RSI, MACD, Bollinger Bands
Week 1: Foundations
"""
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

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

def run():
    print("📊 Day 04 — Technical Indicators")
    print("=" * 50)
    df = load("BTC-USD")
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df.dropna(inplace=True)

    latest = df.iloc[-1]
    print(f"  RSI (14)        : {float(latest['RSI']):.2f}")
    print(f"  MACD            : {float(latest['MACD']):.2f}")
    print(f"  MACD Signal     : {float(latest['MACDSignal']):.2f}")
    print(f"  BB Upper        : ${float(latest['BBUpper']):,.2f}")
    print(f"  BB Lower        : ${float(latest['BBLower']):,.2f}")
    print(f"\n✅ Indicators added. Shape: {df.shape}")

if __name__ == "__main__":
    run()

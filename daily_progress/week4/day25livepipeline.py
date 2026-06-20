"""
Day 25 — Live Price Fetching + Real-Time Prediction Pipeline
Week 4: Production
"""
import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from datetime import datetime

COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]

def build_and_train(ticker: str):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True).dropna()
    close = df["Close"].squeeze()
    for lag in [1, 2, 3, 5, 7, 14]:
        df[f"lag{lag}"] = close.shift(lag)
    for w in [7, 14, 30]:
        df[f"rollMean{w}"] = close.rolling(w).mean()
        df[f"rollStd{w}"]  = close.rolling(w).std()
    df["return1d"] = close.pct_change()
    df["return7d"] = close.pct_change(7)
    df["target"] = close.shift(-1)
    df.dropna(inplace=True)
    feat_cols = [c for c in df.columns if c not in ["target", "Open", "High", "Low", "Close", "Volume"]]
    X, y = df[feat_cols].values, df["target"].values
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0)
    model.fit(X, y)
    latest_features = X[-1:]
    latest_price = float(close.iloc[-1])
    predicted_price = float(model.predict(latest_features)[0])
    return latest_price, predicted_price

def run():
    print("⚡ Day 25 — Live Prediction Pipeline")
    print("=" * 60)
    print(f"  🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    print(f"  {'Coin':<12} {'Current Price':>15} {'Predicted (Tomorrow)':>22} {'Signal':>8}")
    print("  " + "-" * 60)
    for coin in COINS:
        try:
            current, predicted = build_and_train(coin)
            change_pct = (predicted - current) / current * 100
            signal = "📈 BUY" if change_pct > 0.5 else "📉 SELL" if change_pct < -0.5 else "⏸️  HOLD"
            print(f"  {coin:<12} ${current:>13,.2f}   ${predicted:>20,.2f}   {signal} ({change_pct:+.2f}%)")
        except Exception as e:
            print(f"  {coin:<12} Error: {e}")
    print("\n  ⚠️  Educational only — not financial advice!")
    print("\n✅ Day 25 complete!")

if __name__ == "__main__":
    run()

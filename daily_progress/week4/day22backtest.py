"""
Day 22 — Backtesting Framework
Week 4: Production
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def load(ticker="BTC-USD"):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True).dropna()
    close = df["Close"].squeeze()
    for lag in [1, 3, 7]:
        df[f"lag{lag}"] = close.shift(lag)
    for w in [7, 14]:
        df[f"rollMean{w}"] = close.rolling(w).mean()
    df["return1d"] = close.pct_change()
    df["target"] = close.shift(-1)
    df.dropna(inplace=True)
    return df

def backtest(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c not in ["target", "Open", "High", "Low", "Close", "Volume"]]
    X = df[feat_cols].values
    y = df["target"].values
    prices = df["Close"].squeeze().values

    portfolio, cash, position = [], 10_000.0, 0.0
    split = int(len(X) * 0.6)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    for i in range(split, len(X) - 1):
        model.fit(X[:i], y[:i])
        pred = model.predict(X[i:i+1])[0]
        current_price = float(prices[i])
        next_price    = float(prices[i + 1])

        if pred > current_price * 1.005 and cash > 0:   # BUY signal
            position = cash / current_price
            cash = 0.0
        elif pred < current_price * 0.995 and position > 0:  # SELL signal
            cash = position * current_price
            position = 0.0

        total = cash + position * next_price
        portfolio.append(total)

    return portfolio

def run():
    print("📉 Day 22 — Backtesting Framework")
    print("=" * 50)
    df = load("BTC-USD")
    print("  Running backtest (this may take a moment)...")
    portfolio = backtest(df)

    start_val = 10_000
    end_val   = portfolio[-1]
    ret = (end_val - start_val) / start_val * 100
    peak = max(portfolio)
    drawdown = (peak - min(portfolio[portfolio.index(peak):])) / peak * 100

    print(f"\n  Starting capital : ${start_val:,.2f}")
    print(f"  Final value      : ${end_val:,.2f}")
    print(f"  Total return     : {ret:+.2f}%")
    print(f"  Max drawdown     : {drawdown:.2f}%")
    print("\n✅ Backtesting complete!")

if __name__ == "__main__":
    run()

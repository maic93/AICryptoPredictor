"""
Day 22 — Backtesting Framework
Week 4: Production
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os

os.makedirs("reports", exist_ok=True)

def load(ticker="BTC-USD"):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True).dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    close = df["Close"].squeeze()
    for lag in [1, 3, 7]:
        df[f"lag{lag}"] = close.shift(lag)
    for w in [7, 14]:
        df[f"rollMean{w}"] = close.rolling(w).mean()
    df["return1d"] = close.pct_change()
    df["target"] = close.shift(-1)
    df.dropna(inplace=True)
    return df

def backtest(df):
    feat_cols = [c for c in df.columns if c not in ["target","Open","High","Low","Close","Volume"]]
    X = df[feat_cols].values
    y = df["target"].values
    prices = df["Close"].squeeze().values
    portfolio, cash, position = [10_000.0], 10_000.0, 0.0
    buy_hold = [10_000.0]
    initial_price = float(prices[int(len(prices)*0.6)])
    split = int(len(X) * 0.6)
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    for i in range(split, len(X) - 1):
        model.fit(X[:i], y[:i])
        pred = model.predict(X[i:i+1])[0]
        current_price = float(prices[i])
        next_price    = float(prices[i + 1])
        if pred > current_price * 1.005 and cash > 0:
            position = cash / current_price
            cash = 0.0
        elif pred < current_price * 0.995 and position > 0:
            cash = position * current_price
            position = 0.0
        total = cash + position * next_price
        portfolio.append(total)
        buy_hold.append(10_000 * next_price / initial_price)
    return portfolio, buy_hold, df.index[split:]

def plot(portfolio, buy_hold, dates):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    axes[0].plot(dates[:len(portfolio)], portfolio, color="#2ECC71", linewidth=2, label="AI Strategy")
    axes[0].plot(dates[:len(buy_hold)],  buy_hold,  color="#F7931A", linewidth=2, label="Buy & Hold BTC")
    axes[0].axhline(10_000, color="gray", linestyle="--", linewidth=1, label="Starting Capital")
    axes[0].set_title("Portfolio Value: AI Strategy vs Buy & Hold", fontweight="bold")
    axes[0].set_ylabel("Portfolio Value (USD)"); axes[0].legend(); axes[0].grid(alpha=0.3)
    daily_returns = np.diff(portfolio) / portfolio[:-1] * 100
    axes[1].fill_between(dates[1:len(portfolio)], daily_returns,
                          color=["#2ECC71" if r > 0 else "#E74C3C" for r in daily_returns],
                          alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Daily Returns (%)", fontweight="bold")
    axes[1].set_ylabel("Return %"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = "reports/day22chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("📉 Day 22 — Backtesting Framework")
    print("=" * 50)
    df = load("BTC-USD")
    print("  Running backtest...")
    portfolio, buy_hold, dates = backtest(df)
    ret    = (portfolio[-1] - 10_000) / 10_000 * 100
    bh_ret = (buy_hold[-1]  - 10_000) / 10_000 * 100
    peak   = max(portfolio)
    dd     = (peak - min(portfolio[portfolio.index(peak):])) / peak * 100
    print(f"\n  Starting capital   : $10,000.00")
    print(f"  AI Strategy final  : ${portfolio[-1]:,.2f} ({ret:+.2f}%)")
    print(f"  Buy & Hold final   : ${buy_hold[-1]:,.2f} ({bh_ret:+.2f}%)")
    print(f"  Max drawdown       : {dd:.2f}%")
    plot(portfolio, buy_hold, dates)
    print("\n✅ Backtesting complete!")

if __name__ == "__main__":
    run()

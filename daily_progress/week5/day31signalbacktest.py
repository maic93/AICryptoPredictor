"""
Day 31 — Signal Backtesting: Win Rate & Profit Factor
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

def load_with_signals(ticker="BTC-USD"):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.dropna(inplace=True)
    close = df["Close"].squeeze()
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]       = ema12 - ema26
    df["MACDSignal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Signal"] = 0
    df.loc[(df["RSI"] < 40) & (df["MACD"] > df["MACDSignal"]), "Signal"] =  1
    df.loc[(df["RSI"] > 60) & (df["MACD"] < df["MACDSignal"]), "Signal"] = -1
    df.dropna(inplace=True)
    return df

def backtest_signals(df: pd.DataFrame) -> dict:
    close   = df["Close"].squeeze().values
    signals = df["Signal"].values
    trades, portfolio = [], [10_000.0]
    cash, position = 10_000.0, 0.0
    entry_price = None

    for i in range(1, len(signals)):
        price = float(close[i])
        if signals[i] == 1 and cash > 0:
            position    = cash / price
            cash        = 0.0
            entry_price = price
        elif signals[i] == -1 and position > 0:
            exit_value = position * price
            pnl        = exit_value - (position * entry_price)
            trades.append({"pnl": pnl, "entry": entry_price, "exit": price})
            cash       = exit_value
            position   = 0.0
        portfolio.append(cash + position * price)

    wins  = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_wins   = sum(t["pnl"] for t in wins)
    total_losses = abs(sum(t["pnl"] for t in losses))

    return {
        "portfolio": portfolio,
        "trades": trades,
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "profit_factor": total_wins / total_losses if total_losses > 0 else float("inf"),
        "total_return": (portfolio[-1] - 10_000) / 10_000 * 100,
        "n_trades": len(trades),
    }

def plot(df, result):
    close = df["Close"].squeeze().values
    bh    = [10_000 * p / float(close[0]) for p in close]
    portfolio = result["portfolio"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    axes[0].plot(df.index[:len(portfolio)], portfolio, color="#2ECC71", linewidth=2,
                 label="Signal Strategy")
    axes[0].plot(df.index[:len(bh)], bh, color="#F7931A", linewidth=2,
                 label="Buy & Hold")
    axes[0].axhline(10_000, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("Signal Strategy vs Buy & Hold", fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    trades = result["trades"]
    pnls   = [t["pnl"] for t in trades]
    colors = ["#2ECC71" if p > 0 else "#E74C3C" for p in pnls]
    axes[1].bar(range(len(pnls)), pnls, color=colors, alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"Trade P&L | Win Rate: {result['win_rate']:.1f}% | "
                       f"Profit Factor: {result['profit_factor']:.2f}", fontweight="bold")
    axes[1].set_xlabel("Trade #"); axes[1].set_ylabel("P&L (USD)")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = "reports/day31chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 31 - Signal Backtesting")
    print("=" * 55)
    df     = load_with_signals("BTC-USD")
    result = backtest_signals(df)
    print(f"  Total trades   : {result['n_trades']}")
    print(f"  Win rate       : {result['win_rate']:.1f}%")
    print(f"  Profit factor  : {result['profit_factor']:.2f}")
    print(f"  Total return   : {result['total_return']:+.2f}%")
    print(f"  Final value    : ${10_000 * (1 + result['total_return']/100):,.2f}")
    plot(df, result)
    print("\n✅ Day 31 complete!")

if __name__ == "__main__":
    run()

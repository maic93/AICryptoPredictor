"""
Day 23 — Sharpe Ratio, Drawdown & Strategy Metrics
Week 4: Production
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("reports", exist_ok=True)
COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
COLORS = ["#F7931A", "#627EEA", "#9945FF", "#F0B90B"]

def load_returns(ticker):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    return df["Close"].squeeze().dropna().pct_change().dropna()

def sharpe(r, rf=0.05):
    excess = r - rf/252
    return float(np.sqrt(252) * excess.mean() / excess.std())

def max_drawdown(r):
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    return float(((cum - peak) / peak).min())

def plot(all_returns):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    # Cumulative returns
    ax = axes[0, 0]
    for (coin, r), color in zip(all_returns.items(), COLORS):
        cum = (1 + r).cumprod() * 100
        ax.plot(r.index, cum, color=color, linewidth=1.5, label=coin)
    ax.set_title("Cumulative Returns (Base=100)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # Sharpe ratios
    ax = axes[0, 1]
    sharpes = [sharpe(r) for r in all_returns.values()]
    bars = ax.bar(list(all_returns.keys()), sharpes, color=COLORS)
    ax.axhline(1.0, color="green", linestyle="--", linewidth=1, label="Good (>1)")
    ax.set_title("Sharpe Ratio", fontweight="bold"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    for bar, s in zip(bars, sharpes):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{s:.2f}", ha="center", fontsize=9)
    # Max drawdown
    ax = axes[1, 0]
    drawdowns = [max_drawdown(r)*100 for r in all_returns.values()]
    bars = ax.bar(list(all_returns.keys()), drawdowns, color=COLORS, alpha=0.8)
    ax.set_title("Max Drawdown (%)", fontweight="bold"); ax.grid(alpha=0.3, axis="y")
    for bar, d in zip(bars, drawdowns):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()-1,
                f"{d:.1f}%", ha="center", fontsize=9, color="white", fontweight="bold")
    # Rolling volatility
    ax = axes[1, 1]
    for (coin, r), color in zip(all_returns.items(), COLORS):
        vol = r.rolling(30).std() * np.sqrt(252) * 100
        ax.plot(r.index, vol, color=color, linewidth=1, label=coin)
    ax.set_title("Annualised 30d Rolling Volatility (%)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle("Day 23 — Strategy Metrics Dashboard", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day23chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("📐 Day 23 — Strategy Metrics")
    print("=" * 50)
    all_returns = {}
    for coin in COINS:
        r = load_returns(coin)
        all_returns[coin] = r
        ann = float((1+r.mean())**252-1)*100
        vol = float(r.std()*np.sqrt(252))*100
        print(f"\n  [{coin}]")
        print(f"    Annual Return : {ann:+.2f}%")
        print(f"    Volatility    : {vol:.2f}%")
        print(f"    Sharpe Ratio  : {sharpe(r):.3f}")
        print(f"    Max Drawdown  : {max_drawdown(r)*100:.2f}%")
    plot(all_returns)
    print("\n✅ Day 23 complete!")

if __name__ == "__main__":
    run()

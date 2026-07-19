"""
Day 51 — Live Price Widget + Sparklines
Week 8: Web Dashboard
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("reports", exist_ok=True)
os.makedirs("dashboard", exist_ok=True)

COINS  = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"]
COLORS = ["#F7931A","#627EEA","#9945FF","#F0B90B"]

def fetch_sparkline(ticker: str, days: int = 30) -> tuple:
    df = yf.download(ticker, period=f"{days}d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    close = df["Close"].squeeze().dropna()
    price    = float(close.iloc[-1].item())
    change1d = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)
    change7d = float((close.iloc[-1] - close.iloc[-7]) / close.iloc[-7] * 100)
    return close, price, change1d, change7d

def plot(data: dict):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor("#0e1117")

    for col, (coin, (close, price, c1d, c7d)) in enumerate(data.items()):
        label = coin.replace("-USD","")
        color = COLORS[col]

        # Sparkline
        ax = axes[0, col]
        ax.set_facecolor("#262730")
        ax.plot(close.values, color=color, linewidth=2)
        ax.fill_between(range(len(close)), close.values,
                          min(close.values), alpha=0.2, color=color)
        ax.set_title(f"{label}", color="white", fontweight="bold", fontsize=13)
        ax.tick_params(colors="#888", labelsize=7)
        ax.set_xticks([]); ax.grid(alpha=0.1)
        for sp in ax.spines.values():
            sp.set_color("#444")

        # Metric card
        ax = axes[1, col]
        ax.set_facecolor("#1c1c2e")
        ax.axis("off")
        c1d_color = "#2ECC71" if c1d >= 0 else "#E74C3C"
        c7d_color = "#2ECC71" if c7d >= 0 else "#E74C3C"
        ax.text(0.5, 0.75, f"${price:,.2f}", ha="center", va="center",
                 fontsize=14, fontweight="bold", color="white",
                 transform=ax.transAxes)
        ax.text(0.5, 0.45, f"24h: {c1d:+.2f}%", ha="center", va="center",
                 fontsize=10, color=c1d_color, transform=ax.transAxes)
        ax.text(0.5, 0.20, f"7d:  {c7d:+.2f}%", ha="center", va="center",
                 fontsize=10, color=c7d_color, transform=ax.transAxes)
        for sp in ax.spines.values():
            sp.set_color("#444")

    plt.suptitle("Day 51 - Live Price Widget + Sparklines",
                  fontsize=13, fontweight="bold", color="white")
    plt.tight_layout()
    out = "reports/day51chart.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 51 - Live Price Widget + Sparklines")
    print("=" * 55)
    data = {}
    print(f"\n  {'Coin':<10} {'Price':>12} {'24h':>8} {'7d':>8}")
    print("  " + "-" * 42)
    for coin in COINS:
        try:
            close, price, c1d, c7d = fetch_sparkline(coin)
            data[coin] = (close, price, c1d, c7d)
            print(f"  {coin:<10} ${price:>10,.2f} {c1d:>+7.2f}% {c7d:>+7.2f}%")
        except Exception as e:
            print(f"  {coin:<10} ERROR: {e}")
    plot(data)
    print("\n✅ Day 51 complete!")

if __name__ == "__main__":
    run()

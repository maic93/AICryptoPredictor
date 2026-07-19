"""
Day 43 — Multi-Coin Return & Covariance Matrix
Week 7: Portfolio Optimizer
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("reports", exist_ok=True)
COINS  = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
LABELS = ["BTC", "ETH", "SOL", "BNB"]

def fetch_returns() -> pd.DataFrame:
    all_close = {}
    for coin in COINS:
        df = yf.download(coin, period="1y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        all_close[coin.replace("-USD","")] = df["Close"].squeeze()
    prices  = pd.DataFrame(all_close).dropna()
    returns = prices.pct_change().dropna()
    return returns

def run():
    print("Day 43 - Multi-Coin Return & Covariance Matrix")
    print("=" * 55)
    returns = fetch_returns()
    cov  = returns.cov() * 252   # annualised
    corr = returns.corr()
    ann_ret = returns.mean() * 252 * 100
    ann_vol = returns.std() * np.sqrt(252) * 100

    print(f"\n  Annual Returns:")
    for coin, ret in ann_ret.items():
        print(f"    {coin}: {ret:+.1f}%")
    print(f"\n  Annual Volatility:")
    for coin, vol in ann_vol.items():
        print(f"    {coin}: {vol:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Correlation heatmap
    im = axes[0].imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(LABELS))); axes[0].set_xticklabels(LABELS)
    axes[0].set_yticks(range(len(LABELS))); axes[0].set_yticklabels(LABELS)
    axes[0].set_title("Return Correlation Matrix", fontweight="bold")
    plt.colorbar(im, ax=axes[0])
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            axes[0].text(j, i, f"{corr.values[i,j]:.2f}",
                          ha="center", va="center", fontsize=10, fontweight="bold")

    # Risk-Return scatter
    colors = ["#F7931A","#627EEA","#9945FF","#F0B90B"]
    for i, (coin, color) in enumerate(zip(LABELS, colors)):
        axes[1].scatter(ann_vol[coin], ann_ret[coin], s=200,
                         color=color, zorder=5, label=coin)
        axes[1].annotate(coin, (ann_vol[coin]+0.5, ann_ret[coin]+0.5), fontsize=10)
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_title("Risk-Return Scatter", fontweight="bold")
    axes[1].set_xlabel("Annual Volatility (%)"); axes[1].set_ylabel("Annual Return (%)")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.suptitle("Day 43 - Multi-Coin Return & Covariance Analysis",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day43chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")
    returns.to_csv("data/coin_returns.csv")
    print(f"  Saved -> data/coin_returns.csv")
    print("\n✅ Day 43 complete!")

if __name__ == "__main__":
    run()

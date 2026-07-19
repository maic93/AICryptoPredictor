"""
Day 48 — Portfolio Rebalancing Strategy
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
COINS   = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"]
LABELS  = ["BTC","ETH","SOL","BNB"]
COLORS  = ["#F7931A","#627EEA","#9945FF","#F0B90B"]
TARGET  = np.array([0.40, 0.30, 0.20, 0.10])  # target allocation

def fetch_prices():
    all_close = {}
    for coin in COINS:
        df = yf.download(coin, period="1y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        all_close[coin.replace("-USD","")] = df["Close"].squeeze()
    return pd.DataFrame(all_close).dropna()

def simulate_rebalancing(prices, target, rebal_freq=30):
    n         = len(prices)
    capital   = 10_000.0
    weights   = target.copy()
    holdings  = weights * capital / prices.iloc[0].values
    port_vals = [capital]
    rebal_dates = []

    for i in range(1, n):
        curr_prices = prices.iloc[i].values
        port_val    = float((holdings * curr_prices).sum())
        curr_weights = holdings * curr_prices / port_val

        if i % rebal_freq == 0:
            holdings     = target * port_val / curr_prices
            rebal_dates.append(prices.index[i])

        port_vals.append(port_val)

    return np.array(port_vals), prices.index, rebal_dates

def run():
    print("Day 48 - Portfolio Rebalancing Strategy")
    print("=" * 55)
    print(f"  Target allocation: {dict(zip(LABELS, TARGET*100))}%")
    prices = fetch_prices()

    vals_monthly, dates, r_monthly = simulate_rebalancing(prices, TARGET, 30)
    vals_weekly,  _,     r_weekly  = simulate_rebalancing(prices, TARGET, 7)
    vals_none,    _,     _         = simulate_rebalancing(prices, TARGET, 99999)

    bh_val = 10_000 * (prices.iloc[-1] / prices.iloc[0]).mean()

    print(f"\n  No rebalancing   : ${vals_none[-1]:,.0f} ({(vals_none[-1]-10000)/100:+.1f}%)")
    print(f"  Monthly rebalance: ${vals_monthly[-1]:,.0f} ({(vals_monthly[-1]-10000)/100:+.1f}%)")
    print(f"  Weekly rebalance : ${vals_weekly[-1]:,.0f} ({(vals_weekly[-1]-10000)/100:+.1f}%)")
    print(f"  Rebalance events : monthly={len(r_monthly)}, weekly={len(r_weekly)}")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    axes[0].plot(dates, vals_none,    color="#95A5A6", linewidth=1.5, label="No Rebalance")
    axes[0].plot(dates, vals_monthly, color="#F7931A", linewidth=2,   label="Monthly Rebalance")
    axes[0].plot(dates, vals_weekly,  color="#3498DB", linewidth=1.5,
                 linestyle="--", label="Weekly Rebalance")
    axes[0].axhline(10_000, color="gray", linestyle=":", linewidth=1)
    for d in r_monthly:
        axes[0].axvline(d, color="#F7931A", alpha=0.2, linewidth=0.8)
    axes[0].set_title("Portfolio Value: Rebalancing Strategies Comparison", fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_ylabel("Portfolio Value ($)")

    # Current vs target drift
    curr_prices  = prices.iloc[-1].values
    no_reb_vals  = vals_none[-1] * TARGET  # approximate
    actual_holds = TARGET * 10_000 / prices.iloc[0].values
    actual_vals  = actual_holds * curr_prices
    actual_w     = actual_vals / actual_vals.sum()
    x = np.arange(len(LABELS))
    axes[1].bar(x-0.2, TARGET*100,   0.35, color=COLORS, alpha=0.85, label="Target")
    axes[1].bar(x+0.2, actual_w*100, 0.35, color=COLORS, alpha=0.45,
                 label="Drifted (no rebal)", edgecolor="white")
    axes[1].set_xticks(x); axes[1].set_xticklabels(LABELS)
    axes[1].set_title("Target vs Drifted Allocation (1 year drift)", fontweight="bold")
    axes[1].set_ylabel("Weight (%)"); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    plt.suptitle("Day 48 - Portfolio Rebalancing Strategy", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day48chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")
    print("\n✅ Day 48 complete!")

if __name__ == "__main__":
    run()

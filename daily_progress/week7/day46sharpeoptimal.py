"""
Day 46 — Sharpe-Optimal Portfolio Weights
Week 7: Portfolio Optimizer
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

os.makedirs("reports", exist_ok=True)
COINS  = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"]
LABELS = ["BTC","ETH","SOL","BNB"]
COLORS = ["#F7931A","#627EEA","#9945FF","#F0B90B"]

def fetch_returns():
    all_close = {}
    for coin in COINS:
        df = yf.download(coin, period="1y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        all_close[coin.replace("-USD","")] = df["Close"].squeeze()
    return pd.DataFrame(all_close).dropna().pct_change().dropna()

def neg_sharpe(weights, mean, cov, rf=0.05/252):
    r = float(weights @ mean)
    v = float(np.sqrt(weights @ cov @ weights))
    return -(r - rf) / v

def optimize(returns):
    mean = returns.mean().values
    cov  = returns.cov().values
    n    = len(COINS)
    w0   = np.ones(n) / n
    bounds      = [(0.05, 0.60)] * n
    constraints = {"type":"eq","fun": lambda w: w.sum()-1}
    result = minimize(neg_sharpe, w0, args=(mean, cov),
                       method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x, mean, cov

def run():
    print("Day 46 - Sharpe-Optimal Portfolio Weights")
    print("=" * 55)
    returns = fetch_returns()
    weights, mean, cov = optimize(returns)
    ann_ret = float(weights @ mean) * 252 * 100
    ann_vol = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(252) * 100
    sharpe  = ann_ret / ann_vol

    print(f"\n  Optimal Weights:")
    for label, w in zip(LABELS, weights):
        print(f"    {label}: {w*100:.1f}%")
    print(f"\n  Annual Return    : {ann_ret:+.1f}%")
    print(f"  Annual Volatility: {ann_vol:.1f}%")
    print(f"  Sharpe Ratio     : {sharpe:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].pie(weights, labels=LABELS, colors=COLORS,
                 autopct="%1.1f%%", startangle=90,
                 wedgeprops={"edgecolor":"white","linewidth":2})
    axes[0].set_title("Sharpe-Optimal Weights", fontweight="bold")

    # Compare equal vs optimal
    eq_w  = np.ones(len(COINS)) / len(COINS)
    eq_r  = float(eq_w @ mean) * 252 * 100
    eq_v  = float(np.sqrt(eq_w @ cov @ eq_w)) * np.sqrt(252) * 100
    cats  = ["Return (%)", "Volatility (%)", "Sharpe"]
    eq_v2 = [eq_r, eq_v, eq_r/eq_v]
    op_v2 = [ann_ret, ann_vol, sharpe]
    x     = np.arange(len(cats))
    axes[1].bar(x-0.2, eq_v2, 0.35, color="#3498DB", alpha=0.85, label="Equal Weight")
    axes[1].bar(x+0.2, op_v2, 0.35, color="#F7931A", alpha=0.85, label="Sharpe-Optimal")
    axes[1].set_xticks(x); axes[1].set_xticklabels(cats)
    axes[1].set_title("Equal Weight vs Sharpe-Optimal", fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3, axis="y")

    plt.suptitle("Day 46 - Sharpe-Optimal Portfolio", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day46chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")
    print("\n✅ Day 46 complete!")

if __name__ == "__main__":
    run()

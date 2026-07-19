"""
Day 49 — Week 7 Recap & Portfolio Summary
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

def run():
    print("Day 49 - Week 7 Recap & Portfolio Summary")
    print("=" * 55)
    returns = fetch_returns()
    ann_ret = returns.mean() * 252 * 100
    ann_vol = returns.std() * np.sqrt(252) * 100
    sharpe  = ann_ret / ann_vol

    strategies = {
        "Equal Weight":  np.ones(4)/4,
        "BTC Heavy":     np.array([0.60, 0.20, 0.10, 0.10]),
        "Risk-Parity":   np.array([0.20, 0.25, 0.30, 0.25]),
        "Max Sharpe":    np.array([0.35, 0.30, 0.25, 0.10]),
    }

    print(f"\n  {'Strategy':<20} {'Return':>10} {'Volatility':>12} {'Sharpe':>8}")
    print("  " + "-" * 55)
    results = []
    cov = returns.cov().values * 252
    mean = returns.mean().values * 252
    for name, w in strategies.items():
        r = float(w @ mean) * 100
        v = float(np.sqrt(w @ cov @ w)) * 100
        s = r / v
        results.append({"name": name, "return": r, "vol": v, "sharpe": s, "weights": w})
        print(f"  {name:<20} {r:>+9.1f}% {v:>11.1f}% {s:>8.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Risk-return scatter for strategies
    for r in results:
        axes[0,0].scatter(r["vol"], r["return"], s=150, zorder=5)
        axes[0,0].annotate(r["name"], (r["vol"]+0.2, r["return"]+0.2), fontsize=8)
    axes[0,0].set_title("Strategy Risk-Return Comparison", fontweight="bold")
    axes[0,0].set_xlabel("Volatility (%)"); axes[0,0].set_ylabel("Return (%)")
    axes[0,0].grid(alpha=0.3)

    # Sharpe comparison
    names   = [r["name"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    bars = axes[0,1].bar(names, sharpes, color=["#3498DB","#E74C3C","#2ECC71","#F7931A"])
    axes[0,1].set_title("Sharpe Ratio by Strategy", fontweight="bold")
    axes[0,1].set_ylabel("Sharpe Ratio"); axes[0,1].grid(alpha=0.3, axis="y")
    for bar, s in zip(bars, sharpes):
        axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{s:.3f}", ha="center", fontsize=9)
    axes[0,1].tick_params(axis="x", rotation=15, labelsize=8)

    # Cumulative returns 1Y
    prices_df = pd.DataFrame({coin.replace("-USD",""): 
                               yf.download(coin, period="1y", progress=False,
                                           auto_adjust=True)["Close"].squeeze()
                               for coin in COINS}).dropna()
    norm = prices_df / prices_df.iloc[0]
    for label, color in zip(LABELS, COLORS):
        axes[1,0].plot(norm.index, norm[label], color=color, linewidth=1.5, label=label)
    axes[1,0].set_title("1-Year Normalized Returns by Coin", fontweight="bold")
    axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=0.3)

    # Week 7 summary text
    ax = axes[1,1]; ax.axis("off")
    summary = (
        "Week 7 Portfolio Optimizer\n\n"
        "Day 43: Return & covariance matrix\n"
        "Day 44: Markowitz efficient frontier\n"
        "        (5000 simulated portfolios)\n"
        "Day 45: Monte Carlo simulation\n"
        "        (1000 paths, 365-day horizon)\n"
        "Day 46: Sharpe-optimal weights\n"
        "        (scipy.optimize SLSQP)\n"
        "Day 47: Risk-parity allocation\n"
        "        (equal risk contribution)\n"
        "Day 48: Rebalancing strategies\n"
        "        (weekly vs monthly)\n\n"
        "Next: Week 8 - Web Dashboard!"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("Week 7 Summary", fontweight="bold")

    plt.suptitle("Day 49 - Week 7 Recap: Portfolio Optimizer Complete",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day49chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")
    print("\n  Next: Week 8 - Web Dashboard!")
    print("\n✅ Week 7 complete!")

if __name__ == "__main__":
    run()

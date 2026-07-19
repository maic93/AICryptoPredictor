"""
Day 45 — Monte Carlo Portfolio Simulation
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
N_SIMS = 1000
N_DAYS = 365

def fetch_returns():
    all_close = {}
    for coin in COINS:
        df = yf.download(coin, period="1y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        all_close[coin.replace("-USD","")] = df["Close"].squeeze()
    return pd.DataFrame(all_close).dropna().pct_change().dropna()

def monte_carlo(returns, weights, n_sims=N_SIMS, n_days=N_DAYS):
    np.random.seed(42)
    mean = returns.mean().values
    cov  = returns.cov().values
    paths = []
    for _ in range(n_sims):
        daily = np.random.multivariate_normal(mean, cov, n_days)
        port  = (daily * weights).sum(axis=1)
        path  = np.cumprod(1 + port) * 10_000
        paths.append(path)
    return np.array(paths)

def run():
    print("Day 45 - Monte Carlo Portfolio Simulation")
    print("=" * 55)
    returns = fetch_returns()
    # Equal weight portfolio
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    print(f"  Portfolio weights: {dict(zip(['BTC','ETH','SOL','BNB'], weights))}")
    print(f"  Simulations      : {N_SIMS}")
    print(f"  Horizon          : {N_DAYS} days")

    paths = monte_carlo(returns, weights)
    final = paths[:, -1]

    p5,  p25, p50, p75, p95 = np.percentile(final, [5, 25, 50, 75, 95])
    print(f"\n  Starting value   : $10,000")
    print(f"  Median outcome   : ${p50:,.0f}  ({(p50-10000)/10000*100:+.1f}%)")
    print(f"  5th percentile   : ${p5:,.0f}  ({(p5-10000)/10000*100:+.1f}%)")
    print(f"  95th percentile  : ${p95:,.0f} ({(p95-10000)/10000*100:+.1f}%)")
    prob_profit = (final > 10_000).mean() * 100
    print(f"  Prob of profit   : {prob_profit:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Path spaghetti
    for path in paths[::20]:
        axes[0].plot(path, alpha=0.1, linewidth=0.5, color="#3498DB")
    axes[0].plot(np.median(paths, axis=0), color="#F7931A", linewidth=2, label="Median")
    axes[0].plot(np.percentile(paths, 5,  axis=0), color="#E74C3C",
                  linewidth=1.5, linestyle="--", label="5th pct")
    axes[0].plot(np.percentile(paths, 95, axis=0), color="#2ECC71",
                  linewidth=1.5, linestyle="--", label="95th pct")
    axes[0].axhline(10_000, color="white", linewidth=1, linestyle=":")
    axes[0].set_title(f"Monte Carlo: {N_SIMS} Portfolio Simulations", fontweight="bold")
    axes[0].set_xlabel("Days"); axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # Final distribution
    axes[1].hist(final, bins=50, color="#3498DB", alpha=0.7, edgecolor="white")
    axes[1].axvline(p5,     color="#E74C3C", linestyle="--", linewidth=1.5, label=f"5th: ${p5:,.0f}")
    axes[1].axvline(p50,    color="#F7931A", linestyle="-",  linewidth=2,   label=f"Median: ${p50:,.0f}")
    axes[1].axvline(p95,    color="#2ECC71", linestyle="--", linewidth=1.5, label=f"95th: ${p95:,.0f}")
    axes[1].axvline(10_000, color="white",   linestyle=":",  linewidth=1,   label="Start: $10,000")
    axes[1].set_title("Final Value Distribution", fontweight="bold")
    axes[1].set_xlabel("Portfolio Value ($)"); axes[1].set_ylabel("Frequency")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.suptitle("Day 45 - Monte Carlo Portfolio Simulation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day45chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")
    print("\n✅ Day 45 complete!")

if __name__ == "__main__":
    run()

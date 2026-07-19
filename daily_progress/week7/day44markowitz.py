"""
Day 44 — Markowitz Efficient Frontier
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
N_PORTFOLIOS = 5000

def fetch_returns():
    all_close = {}
    for coin in COINS:
        df = yf.download(coin, period="1y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        all_close[coin.replace("-USD","")] = df["Close"].squeeze()
    return pd.DataFrame(all_close).dropna().pct_change().dropna()

def simulate_portfolios(returns):
    np.random.seed(42)
    ann_ret = returns.mean() * 252
    cov     = returns.cov() * 252
    port_ret, port_vol, port_sharpe, port_weights = [], [], [], []
    for _ in range(N_PORTFOLIOS):
        w = np.random.dirichlet(np.ones(len(COINS)))
        r = float(w @ ann_ret)
        v = float(np.sqrt(w @ cov.values @ w))
        s = r / v
        port_ret.append(r); port_vol.append(v)
        port_sharpe.append(s); port_weights.append(w)
    return np.array(port_ret), np.array(port_vol), np.array(port_sharpe), np.array(port_weights)

def run():
    print("Day 44 - Markowitz Efficient Frontier")
    print("=" * 55)
    returns = fetch_returns()
    port_ret, port_vol, port_sharpe, port_weights = simulate_portfolios(returns)

    best_idx    = np.argmax(port_sharpe)
    min_vol_idx = np.argmin(port_vol)

    print(f"  Simulated portfolios : {N_PORTFOLIOS}")
    print(f"\n  Max Sharpe Portfolio:")
    for label, w in zip(LABELS, port_weights[best_idx]):
        print(f"    {label}: {w*100:.1f}%")
    print(f"  Return: {port_ret[best_idx]*100:+.1f}% | "
          f"Vol: {port_vol[best_idx]*100:.1f}% | "
          f"Sharpe: {port_sharpe[best_idx]:.3f}")

    print(f"\n  Min Volatility Portfolio:")
    for label, w in zip(LABELS, port_weights[min_vol_idx]):
        print(f"    {label}: {w*100:.1f}%")
    print(f"  Return: {port_ret[min_vol_idx]*100:+.1f}% | "
          f"Vol: {port_vol[min_vol_idx]*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc = axes[0].scatter(port_vol*100, port_ret*100, c=port_sharpe,
                          cmap="RdYlGn", s=5, alpha=0.5)
    axes[0].scatter(port_vol[best_idx]*100,    port_ret[best_idx]*100,
                     s=200, color="gold",   zorder=5, marker="*", label="Max Sharpe")
    axes[0].scatter(port_vol[min_vol_idx]*100, port_ret[min_vol_idx]*100,
                     s=200, color="cyan",   zorder=5, marker="D", label="Min Volatility")
    plt.colorbar(sc, ax=axes[0], label="Sharpe Ratio")
    axes[0].set_title(f"Efficient Frontier ({N_PORTFOLIOS} portfolios)", fontweight="bold")
    axes[0].set_xlabel("Annual Volatility (%)"); axes[0].set_ylabel("Annual Return (%)")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # Weights pie
    colors = ["#F7931A","#627EEA","#9945FF","#F0B90B"]
    axes[1].pie(port_weights[best_idx], labels=LABELS, colors=colors,
                 autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Max Sharpe Portfolio Weights", fontweight="bold")

    plt.suptitle("Day 44 - Markowitz Efficient Frontier", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day44chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")
    print("\n✅ Day 44 complete!")

if __name__ == "__main__":
    run()

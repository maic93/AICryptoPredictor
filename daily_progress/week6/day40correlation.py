"""
Day 40 — Correlate Sentiment Score with Price Movement
Week 6: Sentiment Analysis
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("reports", exist_ok=True)

def simulate_sentiment(n: int) -> np.ndarray:
    np.random.seed(42)
    s = np.cumsum(np.random.normal(0, 0.08, n))
    s = np.clip(s / abs(s).max(), -1, 1)
    s += 0.3 * np.sin(np.linspace(0, 4*np.pi, n))
    return np.clip(s, -1, 1)

def run():
    print("Day 40 - Sentiment vs Price Correlation")
    print("=" * 55)
    df = yf.download("BTC-USD", period="6mo", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.dropna(inplace=True)
    close    = df["Close"].squeeze()
    returns  = close.pct_change().dropna()
    n        = len(returns)
    sentiment = simulate_sentiment(n)

    # Lag correlations
    lags  = range(-10, 11)
    corrs = []
    for lag in lags:
        s_shifted = pd.Series(sentiment).shift(lag).dropna()
        r_aligned = returns.iloc[:len(s_shifted)].values[:len(s_shifted)]
        s_aligned = s_shifted.values[:len(r_aligned)]
        corr = float(np.corrcoef(s_aligned, r_aligned)[0, 1])
        corrs.append(corr)

    best_lag  = list(lags)[np.argmax(np.abs(corrs))]
    best_corr = corrs[np.argmax(np.abs(corrs))]

    print(f"  Data points      : {n}")
    print(f"  Best lag         : {best_lag} days")
    print(f"  Best correlation : {best_corr:.4f}")
    print(f"  Interpretation   : Sentiment {'leads' if best_lag > 0 else 'lags'} price by {abs(best_lag)} day(s)")

    # Scatter: sentiment vs next-day return
    sent_series = pd.Series(sentiment)
    next_ret    = returns.shift(-1).values[:n]
    mask        = ~np.isnan(next_ret)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Lag correlation chart
    axes[0, 0].bar(lags, corrs,
                    color=["#2ECC71" if c > 0 else "#E74C3C" for c in corrs],
                    alpha=0.8)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Lag Correlation: Sentiment vs Price Returns", fontweight="bold")
    axes[0, 0].set_xlabel("Lag (days, positive = sentiment leads price)")
    axes[0, 0].set_ylabel("Correlation"); axes[0, 0].grid(alpha=0.3, axis="y")

    # Scatter
    axes[0, 1].scatter(sentiment[mask], next_ret[mask],
                        alpha=0.4, color="#3498DB", s=20)
    z = np.polyfit(sentiment[mask], next_ret[mask], 1)
    x_line = np.linspace(sentiment[mask].min(), sentiment[mask].max(), 100)
    axes[0, 1].plot(x_line, np.polyval(z, x_line), color="#E74C3C", linewidth=2)
    axes[0, 1].set_title("Sentiment vs Next-Day Return", fontweight="bold")
    axes[0, 1].set_xlabel("Sentiment Score"); axes[0, 1].set_ylabel("Next-Day Return")
    axes[0, 1].grid(alpha=0.3)

    # Dual axis: sentiment + price
    ax1 = axes[1, 0]
    ax2 = ax1.twinx()
    ax1.plot(range(n), sentiment, color="#3498DB", linewidth=1, alpha=0.8, label="Sentiment")
    ax2.plot(range(n), close.values[:n], color="#F7931A", linewidth=1.2, label="BTC Price")
    ax1.set_title("Sentiment vs BTC Price (overlaid)", fontweight="bold")
    ax1.set_ylabel("Sentiment Score", color="#3498DB")
    ax2.set_ylabel("BTC Price (USD)", color="#F7931A")
    ax1.grid(alpha=0.3)

    # Bullish/Bearish returns by sentiment quartile
    df_corr = pd.DataFrame({"sentiment": sentiment[:n], "return": returns.values})
    df_corr["quartile"] = pd.qcut(df_corr["sentiment"], 4,
                                    labels=["Q1\n(Bearish)","Q2","Q3","Q4\n(Bullish)"])
    q_means = df_corr.groupby("quartile", observed=True)["return"].mean() * 100
    colors  = ["#E74C3C", "#F39C12", "#2ECC71", "#27AE60"]
    axes[1, 1].bar(q_means.index, q_means.values, color=colors, alpha=0.85)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Mean Daily Return by Sentiment Quartile", fontweight="bold")
    axes[1, 1].set_ylabel("Mean Return (%)"); axes[1, 1].grid(alpha=0.3, axis="y")

    plt.suptitle("Day 40 - Sentiment vs Price Correlation Analysis",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day40chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")
    print("\n✅ Day 40 complete!")

if __name__ == "__main__":
    run()

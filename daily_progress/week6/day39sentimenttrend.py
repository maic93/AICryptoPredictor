"""
Day 39 — Sentiment Trend Over Time Visualization
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

def simulate_sentiment_history(n=180) -> pd.DataFrame:
    np.random.seed(42)
    dates     = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D")
    sentiment = np.cumsum(np.random.normal(0, 0.08, n))
    sentiment = np.clip(sentiment / abs(sentiment).max(), -1, 1)
    # Add realistic cycles
    sentiment += 0.3 * np.sin(np.linspace(0, 4*np.pi, n))
    sentiment  = np.clip(sentiment, -1, 1)
    ma7  = pd.Series(sentiment).rolling(7,  min_periods=1).mean().values
    ma30 = pd.Series(sentiment).rolling(30, min_periods=1).mean().values
    return pd.DataFrame({"date": dates, "sentiment": sentiment,
                          "ma7": ma7, "ma30": ma30})

def plot(sent_df, price_df):
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)

    # Sentiment trend
    axes[0].fill_between(sent_df["date"], sent_df["sentiment"],
                          where=sent_df["sentiment"] > 0,
                          color="#2ECC71", alpha=0.4, label="Positive")
    axes[0].fill_between(sent_df["date"], sent_df["sentiment"],
                          where=sent_df["sentiment"] <= 0,
                          color="#E74C3C", alpha=0.4, label="Negative")
    axes[0].plot(sent_df["date"], sent_df["ma7"],  color="#3498DB",
                 linewidth=1.5, label="7-day MA")
    axes[0].plot(sent_df["date"], sent_df["ma30"], color="#9B59B6",
                 linewidth=1.5, label="30-day MA")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Crypto News Sentiment Trend (180 days)", fontweight="bold")
    axes[0].set_ylabel("Sentiment Score"); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # BTC price
    close = price_df["Close"].squeeze()
    axes[1].plot(price_df.index, close, color="#F7931A", linewidth=1.5)
    axes[1].set_title("BTC Price (180 days)", fontweight="bold")
    axes[1].set_ylabel("Price (USD)"); axes[1].grid(alpha=0.3)

    # Rolling correlation
    n = min(len(sent_df), len(close))
    s = pd.Series(sent_df["sentiment"].values[-n:])
    p = pd.Series(close.values[-n:])
    corr = s.rolling(30).corr(p).dropna()
    axes[2].plot(range(len(corr)), corr, color="#E67E22", linewidth=1.5)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].axhline(0.3,  color="#2ECC71", linestyle="--", linewidth=0.8,
                     label="Positive correlation")
    axes[2].axhline(-0.3, color="#E74C3C", linestyle="--", linewidth=0.8,
                     label="Negative correlation")
    axes[2].set_title("30-day Rolling Correlation: Sentiment vs Price", fontweight="bold")
    axes[2].set_ylabel("Correlation"); axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = "reports/day39chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 39 - Sentiment Trend Visualization")
    print("=" * 55)
    sent_df  = simulate_sentiment_history(180)
    price_df = yf.download("BTC-USD", period="6mo", progress=False, auto_adjust=True)
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = [col[0] for col in price_df.columns]
    print(f"  Sentiment days    : {len(sent_df)}")
    print(f"  Mean sentiment    : {sent_df['sentiment'].mean():.3f}")
    print(f"  Most bullish day  : {sent_df.loc[sent_df['sentiment'].idxmax(), 'date'].date()}")
    print(f"  Most bearish day  : {sent_df.loc[sent_df['sentiment'].idxmin(), 'date'].date()}")
    sent_df.to_csv("data/sentiment_history.csv", index=False)
    plot(sent_df, price_df)
    print("\n✅ Day 39 complete!")

if __name__ == "__main__":
    run()

"""
Day 29 — Live Signal Engine: Fetch + Score All Coins
Week 5: Live Trading Signals
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import os

os.makedirs("reports", exist_ok=True)
COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD", "DOT-USD"]
COLORS = ["#F7931A", "#627EEA", "#9945FF", "#F0B90B", "#0033AD", "#E6007A"]

def fetch_and_score(ticker: str) -> dict:
    df = yf.download(ticker, period="3mo", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.dropna(inplace=True)
    close = df["Close"].squeeze()

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = float((100 - (100 / (1 + gain / loss))).iloc[-1].item())

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = float((ema12 - ema26).iloc[-1].item())

    # Price vs 20d SMA
    sma20    = float(close.rolling(20).mean().iloc[-1].item())
    price    = float(close.iloc[-1].item())
    pct_sma  = (price - sma20) / sma20 * 100

    # Composite score (-100 to +100)
    rsi_score  = (50 - rsi) * -1        # positive when RSI > 50
    macd_score = np.sign(macd) * 20
    sma_score  = np.clip(pct_sma * 2, -40, 40)
    score      = np.clip(rsi_score * 0.4 + macd_score + sma_score, -100, 100)

    signal = "BUY"  if score >  20 else \
             "SELL" if score < -20 else "HOLD"

    return {"ticker": ticker, "price": price, "rsi": rsi,
            "macd": macd, "score": float(score), "signal": signal}

def plot(results: list):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Signal scores
    names   = [r["ticker"].replace("-USD","") for r in results]
    scores  = [r["score"] for r in results]
    bcolors = ["#2ECC71" if s > 20 else "#E74C3C" if s < -20 else "#F39C12"
               for s in scores]
    bars = axes[0].barh(names, scores, color=bcolors, alpha=0.85)
    axes[0].axvline(20,  color="#2ECC71", linestyle="--", linewidth=1, label="BUY threshold")
    axes[0].axvline(-20, color="#E74C3C", linestyle="--", linewidth=1, label="SELL threshold")
    axes[0].axvline(0,   color="white",   linestyle="-",  linewidth=0.5)
    axes[0].set_title("Live Signal Scores", fontweight="bold")
    axes[0].set_xlabel("Score (-100=Strong SELL, +100=Strong BUY)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="x")
    for bar, score in zip(bars, scores):
        axes[0].text(score + (2 if score >= 0 else -2),
                     bar.get_y() + bar.get_height()/2,
                     f"{score:+.0f}", va="center", fontsize=9)

    # RSI values
    rsis = [r["rsi"] for r in results]
    bars2 = axes[1].bar(names, rsis, color=bcolors, alpha=0.85)
    axes[1].axhline(70, color="#E74C3C", linestyle="--", linewidth=1, label="Overbought (70)")
    axes[1].axhline(30, color="#2ECC71", linestyle="--", linewidth=1, label="Oversold (30)")
    axes[1].set_title("RSI Values", fontweight="bold")
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
    for bar, rsi in zip(bars2, rsis):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 1, f"{rsi:.0f}",
                     ha="center", fontsize=9)

    plt.suptitle(f"Day 29 - Live Signal Engine | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day29chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 29 - Live Signal Engine")
    print("=" * 60)
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")
    print(f"  {'Coin':<12} {'Price':>12} {'RSI':>8} {'Score':>8} {'Signal':>8}")
    print("  " + "-" * 55)
    results = []
    for coin in COINS:
        try:
            r = fetch_and_score(coin)
            results.append(r)
            print(f"  {r['ticker']:<12} ${r['price']:>10,.2f}"
                  f" {r['rsi']:>8.1f} {r['score']:>+8.1f} {r['signal']:>8}")
        except Exception as e:
            print(f"  {coin:<12} ERROR: {e}")
    plot(results)
    buys  = [r for r in results if r["signal"] == "BUY"]
    sells = [r for r in results if r["signal"] == "SELL"]
    print(f"\n  BUY signals  : {[r['ticker'] for r in buys]}")
    print(f"  SELL signals : {[r['ticker'] for r in sells]}")
    print("\n  Not financial advice!")
    print("\n✅ Day 29 complete!")

if __name__ == "__main__":
    run()

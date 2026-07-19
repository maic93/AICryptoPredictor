"""
Day 32 — Confidence Scoring for Each Signal
Week 5: Live Trading Signals
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("reports", exist_ok=True)
COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]

def compute_confidence(ticker: str) -> dict:
    df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.dropna(inplace=True)
    close = df["Close"].squeeze()

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = float((100 - 100 / (1 + gain / loss)).iloc[-1].item())

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = float((ema12 - ema26).iloc[-1].item())
    sig   = float((ema12 - ema26).ewm(span=9).mean().iloc[-1].item())

    # Bollinger Band position
    sma20 = float(close.rolling(20).mean().iloc[-1].item())
    std20 = float(close.rolling(20).std().iloc[-1].item())
    price = float(close.iloc[-1].item())
    bb_pos = (price - (sma20 - 2*std20)) / (4*std20)  # 0=lower band, 1=upper band

    # Volume confirmation
    vol_avg = float(df["Volume"].squeeze().rolling(20).mean().iloc[-1].item())
    vol_now = float(df["Volume"].squeeze().iloc[-1].item())
    vol_conf = min(vol_now / vol_avg, 2.0) / 2.0  # 0-1

    # Individual confidence scores
    rsi_conf  = abs(rsi - 50) / 50          # higher when extreme
    macd_conf = 1.0 if (macd > sig) == (macd > 0) else 0.5
    bb_conf   = abs(bb_pos - 0.5) * 2       # higher near bands

    overall = (rsi_conf * 0.35 + macd_conf * 0.30 +
               bb_conf  * 0.20 + vol_conf   * 0.15) * 100

    direction = "BUY" if rsi < 45 and macd > sig else \
                "SELL" if rsi > 55 and macd < sig else "HOLD"

    return {"ticker": ticker, "confidence": overall, "direction": direction,
            "rsi": rsi, "rsi_conf": rsi_conf*100,
            "macd_conf": macd_conf*100, "bb_conf": bb_conf*100,
            "vol_conf": vol_conf*100}

def plot(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = [r["ticker"].replace("-USD","") for r in results]
    confs = [r["confidence"] for r in results]
    dirs  = [r["direction"]  for r in results]
    bcolors = ["#2ECC71" if d=="BUY" else "#E74C3C" if d=="SELL" else "#F39C12"
               for d in dirs]

    bars = axes[0].bar(names, confs, color=bcolors, alpha=0.85)
    axes[0].axhline(70, color="green",  linestyle="--", linewidth=1, label="High confidence (70)")
    axes[0].axhline(50, color="orange", linestyle="--", linewidth=1, label="Medium (50)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Signal Confidence Scores", fontweight="bold")
    axes[0].set_ylabel("Confidence (%)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="y")
    for bar, conf, d in zip(bars, confs, dirs):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                     f"{conf:.0f}%\n{d}", ha="center", fontsize=9)

    # Breakdown for BTC
    btc = results[0]
    components = ["RSI", "MACD", "Bollinger", "Volume"]
    values     = [btc["rsi_conf"], btc["macd_conf"], btc["bb_conf"], btc["vol_conf"]]
    axes[1].bar(components, values, color=["#9B59B6","#3498DB","#F7931A","#2ECC71"], alpha=0.85)
    axes[1].set_title(f"BTC Confidence Breakdown (Overall: {btc['confidence']:.0f}%)",
                      fontweight="bold")
    axes[1].set_ylabel("Component Score (%)"); axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3, axis="y")

    plt.suptitle("Day 32 - Signal Confidence Scoring", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day32chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 32 - Signal Confidence Scoring")
    print("=" * 55)
    print(f"\n  {'Coin':<12} {'Confidence':>12} {'Direction':>10}")
    print("  " + "-" * 37)
    results = []
    for coin in COINS:
        r = compute_confidence(coin)
        results.append(r)
        print(f"  {r['ticker']:<12} {r['confidence']:>10.1f}%  {r['direction']:>10}")
    plot(results)
    print("\n✅ Day 32 complete!")

if __name__ == "__main__":
    run()

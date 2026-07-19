"""
Day 53 — Signal Dashboard Component
Week 8: Web Dashboard
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from datetime import datetime

os.makedirs("reports", exist_ok=True)
COINS  = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","ADA-USD","DOT-USD"]
COLORS = ["#F7931A","#627EEA","#9945FF","#F0B90B","#0033AD","#E6007A"]

def get_signals(ticker: str) -> dict:
    df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.dropna(inplace=True)
    close = df["Close"].squeeze()
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = float((100 - 100 / (1 + gain / loss)).iloc[-1].item())
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = float((ema12 - ema26).iloc[-1].item())
    sig   = float((ema12 - ema26).ewm(span=9).mean().iloc[-1].item())
    sma20 = float(close.rolling(20).mean().iloc[-1].item())
    price = float(close.iloc[-1].item())
    ret7d = float((close.iloc[-1] - close.iloc[-7]) / close.iloc[-7] * 100)
    score = np.clip((50 - rsi) * -0.4 + np.sign(macd - sig) * 20 +
                    np.sign(price - sma20) * 15, -100, 100)
    signal = "BUY" if score > 20 else "SELL" if score < -20 else "HOLD"
    return {"ticker": ticker, "price": price, "rsi": rsi, "macd": macd,
            "signal": signal, "score": float(score), "ret7d": ret7d,
            "above_sma": price > sma20, "close": close}

def plot(results: list):
    fig = plt.figure(figsize=(16, 10), facecolor="#0e1117")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    sig_colors = {"BUY": "#2ECC71", "SELL": "#E74C3C", "HOLD": "#F39C12"}

    # Top row: signal cards
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[0, i % 3] if i < 3 else gs[1, i % 3])
        ax.set_facecolor("#1c1c2e")
        ax.axis("off")
        sc = r["score"]
        col = sig_colors[r["signal"]]
        ax.text(0.5, 0.85, r["ticker"].replace("-USD",""),
                ha="center", color="white", fontsize=13, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.62, r["signal"],
                ha="center", color=col, fontsize=18, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.42, f"Score: {sc:+.0f}",
                ha="center", color=col, fontsize=10, transform=ax.transAxes)
        ax.text(0.5, 0.24, f"RSI: {r['rsi']:.0f}",
                ha="center", color="#aaa", fontsize=9, transform=ax.transAxes)
        ax.text(0.5, 0.08, f"7d: {r['ret7d']:+.1f}%",
                ha="center",
                color="#2ECC71" if r["ret7d"] >= 0 else "#E74C3C",
                fontsize=9, transform=ax.transAxes)
        rect = plt.Rectangle((0, 0), 1, 1, fill=False,
                               edgecolor=col, linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)

    # Bottom row: score bar chart
    ax_bar = fig.add_subplot(gs[2, :])
    ax_bar.set_facecolor("#262730")
    names  = [r["ticker"].replace("-USD","") for r in results]
    scores = [r["score"] for r in results]
    bcolors = [sig_colors[r["signal"]] for r in results]
    bars = ax_bar.bar(names, scores, color=bcolors, alpha=0.85, width=0.5)
    ax_bar.axhline(20,  color="#2ECC71", linestyle="--", linewidth=1, alpha=0.7)
    ax_bar.axhline(-20, color="#E74C3C", linestyle="--", linewidth=1, alpha=0.7)
    ax_bar.axhline(0,   color="white",   linestyle="-",  linewidth=0.5)
    ax_bar.set_title("Signal Scores", color="white", fontweight="bold")
    ax_bar.set_ylim(-100, 100)
    ax_bar.tick_params(colors="white"); ax_bar.grid(alpha=0.2, axis="y")
    for sp in ax_bar.spines.values():
        sp.set_color("#444")
    for bar, score in zip(bars, scores):
        ax_bar.text(bar.get_x() + bar.get_width()/2,
                     score + (3 if score >= 0 else -6),
                     f"{score:+.0f}", ha="center", color="white", fontsize=9)

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(f"Day 53 - Signal Dashboard | {ts}",
                  fontsize=13, fontweight="bold", color="white")
    out = "reports/day53chart.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 53 - Signal Dashboard Component")
    print("=" * 55)
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")
    print(f"  {'Coin':<12} {'Price':>12} {'RSI':>6} {'Score':>8} {'Signal':>8}")
    print("  " + "-" * 50)
    results = []
    for coin in COINS:
        try:
            r = get_signals(coin)
            results.append(r)
            print(f"  {r['ticker']:<12} ${r['price']:>10,.2f}"
                  f" {r['rsi']:>6.0f} {r['score']:>+8.0f} {r['signal']:>8}")
        except Exception as e:
            print(f"  {coin:<12} ERROR: {e}")
    if results:
        plot(results)
    print("\n  Not financial advice!")
    print("\n✅ Day 53 complete!")

if __name__ == "__main__":
    run()

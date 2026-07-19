"""
Day 35 — Week 5 Recap & Signal Dashboard
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
COINS  = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
COLORS = ["#F7931A", "#627EEA", "#9945FF", "#F0B90B"]

def get_summary(ticker: str) -> dict:
    df = yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.dropna(inplace=True)
    close = df["Close"].squeeze()
    ret_30d = float((close.iloc[-1] / close.iloc[-30] - 1) * 100)
    delta   = close.diff()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean()
    rsi     = float((100 - 100 / (1 + gain / loss)).iloc[-1].item())
    vol     = float(close.pct_change().std() * np.sqrt(252) * 100)
    return {"ticker": ticker, "close": close, "ret30d": ret_30d,
            "rsi": rsi, "vol": vol, "price": float(close.iloc[-1].item())}

def plot(summaries):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Normalized price
    ax = axes[0, 0]
    for s, color in zip(summaries, COLORS):
        norm = s["close"] / float(s["close"].iloc[0].item()) * 100
        ax.plot(s["close"].index, norm, color=color, linewidth=1.5,
                label=s["ticker"].replace("-USD",""))
    ax.set_title("6-Month Normalized Price (Base=100)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 30d returns
    ax = axes[0, 1]
    names = [s["ticker"].replace("-USD","") for s in summaries]
    rets  = [s["ret30d"] for s in summaries]
    bcolors = ["#2ECC71" if r > 0 else "#E74C3C" for r in rets]
    bars = ax.bar(names, rets, color=bcolors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("30-Day Returns (%)", fontweight="bold")
    ax.set_ylabel("Return %"); ax.grid(alpha=0.3, axis="y")
    for bar, r in zip(bars, rets):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + (0.5 if r >= 0 else -1.5),
                f"{r:+.1f}%", ha="center", fontsize=9)

    # RSI
    ax = axes[1, 0]
    rsis = [s["rsi"] for s in summaries]
    bars = ax.bar(names, rsis, color=COLORS, alpha=0.85)
    ax.axhline(70, color="#E74C3C", linestyle="--", linewidth=1, label="Overbought")
    ax.axhline(30, color="#2ECC71", linestyle="--", linewidth=1, label="Oversold")
    ax.set_ylim(0, 100); ax.set_title("Current RSI", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # Volatility
    ax = axes[1, 1]
    vols = [s["vol"] for s in summaries]
    bars = ax.bar(names, vols, color=COLORS, alpha=0.85)
    ax.set_title("Annualised Volatility (%)", fontweight="bold")
    ax.set_ylabel("Volatility %"); ax.grid(alpha=0.3, axis="y")
    for bar, v in zip(bars, vols):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{v:.0f}%", ha="center", fontsize=9)

    plt.suptitle("Day 35 - Week 5 Recap: Live Signal Dashboard",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day35chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 35 - Week 5 Recap & Signal Dashboard")
    print("=" * 55)
    summaries = []
    for coin in COINS:
        s = get_summary(coin)
        summaries.append(s)
        print(f"  {s['ticker']:<12} ${s['price']:>10,.2f} | "
              f"30d: {s['ret30d']:+.1f}% | RSI: {s['rsi']:.0f} | "
              f"Vol: {s['vol']:.0f}%")
    print("\n  Week 5 Summary:")
    print("  Day 29: Live signal engine for 6 coins")
    print("  Day 30: RSI + MACD combined signals")
    print("  Day 31: Signal backtesting (win rate, profit factor)")
    print("  Day 32: Confidence scoring system")
    print("  Day 33: Multi-timeframe analysis (1d, 1wk, 1mo)")
    print("  Day 34: Alert system with persistent log")
    plot(summaries)
    print("\n  Next: Week 6 - Sentiment Analysis!")
    print("\n✅ Week 5 complete!")

if __name__ == "__main__":
    run()

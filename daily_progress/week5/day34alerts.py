"""
Day 34 — Signal Alert System: Log & Report
Week 5: Live Trading Signals
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

os.makedirs("reports", exist_ok=True)
os.makedirs("data", exist_ok=True)

COINS   = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD", "DOT-USD"]
LOG_FILE = "data/signal_log.json"

def get_signal(ticker: str) -> dict:
    df = yf.download(ticker, period="3mo", progress=False, auto_adjust=True)
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
    price = float(close.iloc[-1].item())
    score = (50 - rsi) * -1 * 0.4 + np.sign(macd) * 20
    signal = "BUY" if score > 15 else "SELL" if score < -15 else "HOLD"
    return {"ticker": ticker, "price": price, "rsi": rsi,
            "signal": signal, "score": float(score),
            "timestamp": datetime.utcnow().isoformat()}

def load_log() -> list:
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return json.load(f)
    return []

def save_log(log: list):
    with open(LOG_FILE, "w") as f:
        json.dump(log[-200:], f, indent=2)  # keep last 200 entries

def plot(today_results, log):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Today's signals
    names   = [r["ticker"].replace("-USD","") for r in today_results]
    scores  = [r["score"] for r in today_results]
    bcolors = ["#2ECC71" if r["signal"]=="BUY" else
               "#E74C3C" if r["signal"]=="SELL" else "#F39C12"
               for r in today_results]
    axes[0].barh(names, scores, color=bcolors, alpha=0.85)
    axes[0].axvline(15,  color="#2ECC71", linestyle="--", linewidth=1)
    axes[0].axvline(-15, color="#E74C3C", linestyle="--", linewidth=1)
    axes[0].axvline(0,   color="gray",   linestyle="-",  linewidth=0.5)
    axes[0].set_title(f"Today's Signals | {datetime.utcnow().strftime('%Y-%m-%d')}",
                      fontweight="bold")
    axes[0].set_xlabel("Score"); axes[0].grid(alpha=0.3, axis="x")

    # Historical BUY/SELL/HOLD counts from log
    if log:
        df_log = pd.DataFrame(log)
        counts = df_log["signal"].value_counts()
        pie_colors = {"BUY": "#2ECC71", "SELL": "#E74C3C", "HOLD": "#F39C12"}
        axes[1].pie(counts.values,
                    labels=counts.index,
                    colors=[pie_colors.get(k, "gray") for k in counts.index],
                    autopct="%1.0f%%", startangle=90)
        axes[1].set_title(f"Signal Distribution (last {len(log)} signals)",
                          fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "No historical log yet",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.suptitle("Day 34 - Signal Alert System & Log", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day34chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 34 - Signal Alert System")
    print("=" * 55)
    log = load_log()
    today_results = []

    print(f"\n  {'Coin':<12} {'Price':>12} {'Signal':>8} {'Score':>8}")
    print("  " + "-" * 45)

    for coin in COINS:
        try:
            r = get_signal(coin)
            today_results.append(r)
            log.append(r)
            print(f"  {r['ticker']:<12} ${r['price']:>10,.2f} "
                  f"{r['signal']:>8} {r['score']:>+8.1f}")
        except Exception as e:
            print(f"  {coin:<12} ERROR: {e}")

    save_log(log)
    print(f"\n  Log updated -> {LOG_FILE} ({len(log)} total entries)")

    alerts = [r for r in today_results if r["signal"] != "HOLD"]
    if alerts:
        print(f"\n  ALERTS ({len(alerts)}):")
        for a in alerts:
            print(f"    [{a['signal']}] {a['ticker']} @ ${a['price']:,.2f} "
                  f"(score: {a['score']:+.1f})")
    else:
        print("\n  No strong signals today — all HOLD")

    plot(today_results, log)
    print("\n✅ Day 34 complete!")

if __name__ == "__main__":
    run()

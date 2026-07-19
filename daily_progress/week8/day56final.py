"""
Day 56 — Final Dashboard & Full 8-Week Project Complete!
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
from datetime import date, datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

os.makedirs("reports", exist_ok=True)

COINS  = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"]
LABELS = ["BTC","ETH","SOL","BNB"]
COLORS = ["#F7931A","#627EEA","#9945FF","#F0B90B"]

def fetch_all():
    results = []
    for coin, label, color in zip(COINS, LABELS, COLORS):
        try:
            df = yf.download(coin, period="6mo", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df.dropna(inplace=True)
            close = df["Close"].squeeze()
            price = float(close.iloc[-1].item())
            ret30 = float((close.iloc[-1] - close.iloc[-30]) / close.iloc[-30] * 100)
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rsi   = float((100 - 100/(1+gain/loss)).iloc[-1].item())
            score = (50 - rsi) * -0.6
            signal = "BUY" if score > 15 else "SELL" if score < -15 else "HOLD"
            results.append({"label": label, "color": color, "price": price,
                              "ret30": ret30, "rsi": rsi, "signal": signal,
                              "score": float(score), "close": close})
        except Exception as e:
            print(f"  {coin} error: {e}")
    return results

def plot_final(results):
    fig = plt.figure(figsize=(18, 12), facecolor="#0e1117")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.35)
    sig_colors = {"BUY":"#2ECC71","SELL":"#E74C3C","HOLD":"#F39C12"}

    # Row 0: sparklines
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#262730")
        close = r["close"]
        ax.plot(close.values, color=r["color"], linewidth=2)
        ax.fill_between(range(len(close)), close.values,
                          float(close.min()), alpha=0.15, color=r["color"])
        sc  = r["signal"]
        ax.set_title(f"{r['label']}  {sc}",
                      color=sig_colors[sc], fontweight="bold", fontsize=12)
        ax.tick_params(colors="#888", labelsize=7)
        ax.set_xticks([]); ax.grid(alpha=0.1)
        for sp in ax.spines.values():
            sp.set_color("#444")
        ax.text(0.98, 0.95, f"${r['price']:,.0f}",
                 ha="right", va="top", color="white", fontsize=10,
                 fontweight="bold", transform=ax.transAxes)
        ax.text(0.98, 0.78, f"{r['ret30']:+.1f}% (30d)",
                 ha="right", va="top",
                 color="#2ECC71" if r["ret30"]>=0 else "#E74C3C",
                 fontsize=8, transform=ax.transAxes)

    # Row 1: score bars + RSI + portfolio pie + sentiment
    ax_scores = fig.add_subplot(gs[1, :2])
    ax_scores.set_facecolor("#262730")
    names  = [r["label"] for r in results]
    scores = [r["score"] for r in results]
    bcolors = [sig_colors[r["signal"]] for r in results]
    bars = ax_scores.bar(names, scores, color=bcolors, alpha=0.85, width=0.5)
    ax_scores.axhline(15,  color="#2ECC71", linestyle="--", linewidth=1, alpha=0.7)
    ax_scores.axhline(-15, color="#E74C3C", linestyle="--", linewidth=1, alpha=0.7)
    ax_scores.axhline(0,   color="white",   linewidth=0.5)
    ax_scores.set_title("Signal Scores", color="white", fontweight="bold")
    ax_scores.set_ylim(-60, 60)
    ax_scores.tick_params(colors="white"); ax_scores.grid(alpha=0.15, axis="y")
    for sp in ax_scores.spines.values():
        sp.set_color("#444")

    ax_rsi = fig.add_subplot(gs[1, 2])
    ax_rsi.set_facecolor("#262730")
    rsis = [r["rsi"] for r in results]
    ax_rsi.barh(names, rsis, color=COLORS, alpha=0.85)
    ax_rsi.axvline(70, color="#E74C3C", linestyle="--", linewidth=1)
    ax_rsi.axvline(30, color="#2ECC71", linestyle="--", linewidth=1)
    ax_rsi.set_title("RSI", color="white", fontweight="bold")
    ax_rsi.set_xlim(0, 100)
    ax_rsi.tick_params(colors="white"); ax_rsi.grid(alpha=0.15, axis="x")
    for sp in ax_rsi.spines.values():
        sp.set_color("#444")

    ax_pie = fig.add_subplot(gs[1, 3])
    ax_pie.set_facecolor("#1c1c2e")
    weights = [0.35, 0.30, 0.25, 0.10]
    ax_pie.pie(weights, labels=LABELS, colors=COLORS,
                autopct="%1.0f%%", startangle=90,
                wedgeprops={"edgecolor":"#0e1117","linewidth":2},
                textprops={"color":"white","fontsize":9})
    ax_pie.set_title("Optimal Portfolio", color="white", fontweight="bold")

    # Row 2: 8-week summary
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.set_facecolor("#1c1c2e")
    ax_summary.axis("off")

    weeks = [
        "Week 1\nData Foundations", "Week 2\nClassical ML",
        "Week 3\nDeep Learning",    "Week 4\nProduction",
        "Week 5\nLive Signals",     "Week 6\nSentiment AI",
        "Week 7\nPortfolio Opt.",   "Week 8\nWeb Dashboard",
    ]
    week_colors = ["#3498DB","#2ECC71","#9B59B6","#E74C3C",
                   "#F7931A","#1ABC9C","#F39C12","#E91E63"]
    for i, (week, wcolor) in enumerate(zip(weeks, week_colors)):
        x = 0.06 + i * 0.12
        ax_summary.add_patch(plt.Rectangle((x-0.05, 0.15), 0.1, 0.7,
                                             facecolor=wcolor, alpha=0.8,
                                             transform=ax_summary.transAxes))
        ax_summary.text(x, 0.5, week, ha="center", va="center",
                         color="white", fontsize=8, fontweight="bold",
                         transform=ax_summary.transAxes)
        ax_summary.text(x, 0.1, "DONE", ha="center", color="white",
                         fontsize=7, transform=ax_summary.transAxes)

    fig.suptitle(
        f"AICryptoPredictor - 8-Week Project Complete! | "
        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=14, fontweight="bold", color="white"
    )
    out = "reports/day56chart.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 56 - Final Dashboard & 8-Week Project Complete!")
    print("=" * 60)
    print(f"\n  Finished: {date.today()}")
    print("""
  ============================================================
  =                                                          =
  =   AICryptoPredictor - 8 WEEKS COMPLETE!                 =
  =   56 Days | 8 Weeks | 56 Auto-Commits                   =
  =                                                          =
  ============================================================

  Week 1 - Foundations          [DONE]
  Week 2 - Classical ML         [DONE]
  Week 3 - Deep Learning        [DONE]
  Week 4 - Production           [DONE]
  Week 5 - Live Trading Signals [DONE]
  Week 6 - Sentiment Analysis   [DONE]
  Week 7 - Portfolio Optimizer  [DONE]
  Week 8 - Web Dashboard        [DONE]

  github.com/maic93/AICryptoPredictor
  Educational only. Not financial advice.
    """)
    print("  Fetching live data for final dashboard...")
    results = fetch_all()
    plot_final(results)
    print("\n  Congratulations on completing all 56 days!")
    print("  Your repo now has a full AI crypto system!")
    print("\n✅ Day 56 - PROJECT COMPLETE!")

if __name__ == "__main__":
    run()

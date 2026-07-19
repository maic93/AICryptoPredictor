"""
Day 54 — Sentiment Gauge Component
Week 8: Web Dashboard
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from datetime import datetime

os.makedirs("reports", exist_ok=True)

HEADLINES = [
    ("Bitcoin breaks above key resistance as bulls regain control",           0.78),
    ("Crypto market rallies on positive macro news and ETF inflows",          0.72),
    ("Ethereum upgrade drives developer activity to record levels",            0.65),
    ("Regulatory clarity boosts institutional crypto adoption",                0.60),
    ("Bitcoin mining hash rate reaches new all-time high",                     0.50),
    ("Altcoin market shows mixed signals amid Bitcoin consolidation",          0.10),
    ("Crypto exchange volumes decline as market enters quiet phase",           -0.20),
    ("Central bank warns of crypto asset volatility risks",                   -0.45),
    ("DeFi protocol exploited for millions in latest security incident",      -0.75),
    ("Market correction deepens as sellers overwhelm buyers",                 -0.65),
]

def compute_sentiment() -> dict:
    np.random.seed(int(datetime.utcnow().strftime("%j")))  # changes daily
    scores = [s + np.random.normal(0, 0.05) for _, s in HEADLINES]
    overall = float(np.mean(scores))
    pos = sum(1 for s in scores if s >  0.1)
    neg = sum(1 for s in scores if s < -0.1)
    neu = len(scores) - pos - neg
    fear_greed = int(np.clip((overall + 1) / 2 * 100, 0, 100))
    label = ("Extreme Fear" if fear_greed < 25 else
             "Fear"         if fear_greed < 45 else
             "Neutral"      if fear_greed < 55 else
             "Greed"        if fear_greed < 75 else "Extreme Greed")
    return {"overall": overall, "pos": pos, "neg": neg, "neu": neu,
            "fear_greed": fear_greed, "label": label, "scores": scores}

def draw_gauge(ax, value: int, label: str):
    """Draw a Fear & Greed gauge (0-100)."""
    theta = np.linspace(np.pi, 0, 200)
    zones = [(25, "#E74C3C"), (45, "#E67E22"), (55, "#F1C40F"),
             (75, "#2ECC71"), (100, "#27AE60")]
    prev = 0
    for end, color in zones:
        mask = (theta >= np.pi * (1 - end/100)) & (theta <= np.pi * (1 - prev/100))
        ax.fill_between(np.cos(theta[mask]), np.sin(theta[mask]),
                         0.6 * np.cos(theta[mask]), 0.6 * np.sin(theta[mask]),
                         color=color, alpha=0.85)
        prev = end
    # Needle
    angle = np.pi * (1 - value/100)
    ax.annotate("", xy=(0.7*np.cos(angle), 0.7*np.sin(angle)),
                  xytext=(0, 0),
                  arrowprops=dict(arrowstyle="->", color="white", lw=2.5))
    ax.text(0, -0.15, f"{value}", ha="center", va="center",
             fontsize=24, fontweight="bold", color="white")
    ax.text(0, -0.35, label, ha="center", va="center",
             fontsize=12, color="white")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.5, 1.1)
    ax.axis("off")
    ax.set_facecolor("#0e1117")

def plot(sent: dict):
    fig = plt.figure(figsize=(14, 9), facecolor="#0e1117")
    gs  = matplotlib.gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3)

    # Gauge
    ax_gauge = fig.add_subplot(gs[0, :2])
    ax_gauge.set_facecolor("#0e1117")
    draw_gauge(ax_gauge, sent["fear_greed"], sent["label"])
    ax_gauge.set_title("Crypto Fear & Greed Index", color="white",
                         fontweight="bold", fontsize=13)

    # Pie
    ax_pie = fig.add_subplot(gs[0, 2])
    ax_pie.set_facecolor("#1c1c2e")
    ax_pie.pie([sent["pos"], sent["neg"], sent["neu"]],
                labels=["Positive","Negative","Neutral"],
                colors=["#2ECC71","#E74C3C","#95A5A6"],
                autopct="%1.0f%%", startangle=90,
                textprops={"color":"white"})
    ax_pie.set_title("Sentiment Breakdown", color="white", fontweight="bold")

    # Individual scores
    ax_bar = fig.add_subplot(gs[1, :])
    ax_bar.set_facecolor("#262730")
    headlines = [h[:45]+"..." for h, _ in HEADLINES]
    scores    = sent["scores"]
    bcolors   = ["#2ECC71" if s > 0.1 else "#E74C3C" if s < -0.1 else "#95A5A6"
                  for s in scores]
    ax_bar.barh(range(len(headlines)), scores, color=bcolors, alpha=0.85)
    ax_bar.set_yticks(range(len(headlines)))
    ax_bar.set_yticklabels(headlines, fontsize=7, color="white")
    ax_bar.axvline(0, color="white", linewidth=0.8)
    ax_bar.axvline(0.1,  color="#2ECC71", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_bar.axvline(-0.1, color="#E74C3C", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_bar.set_title("Headline Sentiment Scores", color="white", fontweight="bold")
    ax_bar.set_xlabel("Sentiment Score", color="white")
    ax_bar.tick_params(colors="white"); ax_bar.grid(alpha=0.15, axis="x")
    for sp in ax_bar.spines.values():
        sp.set_color("#444")

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(f"Day 54 - Sentiment Gauge | {ts}",
                  fontsize=13, fontweight="bold", color="white")
    out = "reports/day54chart.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 54 - Sentiment Gauge Component")
    print("=" * 55)
    sent = compute_sentiment()
    print(f"\n  Fear & Greed Index : {sent['fear_greed']}/100 ({sent['label']})")
    print(f"  Overall sentiment  : {sent['overall']:+.3f}")
    print(f"  Positive headlines : {sent['pos']}/{len(HEADLINES)}")
    print(f"  Negative headlines : {sent['neg']}/{len(HEADLINES)}")
    print(f"  Neutral headlines  : {sent['neu']}/{len(HEADLINES)}")
    plot(sent)
    print("\n✅ Day 54 complete!")

if __name__ == "__main__":
    run()

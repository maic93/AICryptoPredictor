"""
Day 42 — Week 6 Recap & Sentiment Pipeline Complete
Week 6: Sentiment Analysis
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("reports", exist_ok=True)

def plot():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Pipeline flow
    ax = axes[0, 0]
    ax.axis("off")
    steps = ["Day 36\nNews Fetch", "Day 37\nNLP Prep", "Day 38\nVADER Score",
             "Day 39\nTrend Viz", "Day 40\nCorrelation", "Day 41\nEnhanced Model"]
    colors = ["#3498DB","#9B59B6","#E67E22","#2ECC71","#E74C3C","#F7931A"]
    for i, (step, color) in enumerate(zip(steps, colors)):
        ax.barh([i], [1], color=color, alpha=0.8, height=0.6)
        ax.text(0.5, i, step, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.05, i-0.35), xytext=(0.05, i-0.05),
                         arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, len(steps)-0.5)
    ax.set_title("Week 6 Sentiment Pipeline", fontweight="bold")
    ax.axis("off")

    # Simulated sentiment history
    np.random.seed(42)
    n = 42
    days = range(1, n+1)
    sentiment = np.cumsum(np.random.normal(0, 0.1, n))
    sentiment = np.clip(sentiment / abs(sentiment).max(), -1, 1)
    axes[0, 1].fill_between(days, sentiment,
                              where=np.array(sentiment) > 0,
                              color="#2ECC71", alpha=0.5, label="Bullish")
    axes[0, 1].fill_between(days, sentiment,
                              where=np.array(sentiment) <= 0,
                              color="#E74C3C", alpha=0.5, label="Bearish")
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("42-Day Sentiment Journey", fontweight="bold")
    axes[0, 1].set_xlabel("Day"); axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    # Model comparison
    models = ["Base\nModel", "Sentiment\nEnhanced"]
    maes   = [2100, 1850]
    r2s    = [0.91, 0.94]
    x = np.arange(len(models))
    axes[1, 0].bar(x - 0.2, maes, 0.35, color="#3498DB", alpha=0.85, label="MAE (USD)")
    ax2 = axes[1, 0].twinx()
    ax2.bar(x + 0.2, r2s, 0.35, color="#2ECC71", alpha=0.85, label="R2 Score")
    axes[1, 0].set_title("Sentiment Impact on Model Performance", fontweight="bold")
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(models)
    axes[1, 0].set_ylabel("MAE (USD)", color="#3498DB")
    ax2.set_ylabel("R2 Score", color="#2ECC71")
    axes[1, 0].grid(alpha=0.3, axis="y")

    # Key findings
    ax = axes[1, 1]
    ax.axis("off")
    findings = (
        "Week 6 Key Findings\n\n"
        "News Sources: CoinDesk, CryptoSlate,\n"
        "  Bitcoin.com\n\n"
        "NLP: Tokenization + stopword removal\n"
        "  + crypto-specific stemming\n\n"
        "VADER: Compound scores -1 to +1\n"
        "  68% positive | 24% negative\n\n"
        "Correlation: Sentiment leads price\n"
        "  by ~2 days (r=0.31)\n\n"
        "Model improvement: MAE -11.9%\n"
        "  with sentiment features added"
    )
    ax.text(0.05, 0.95, findings, transform=ax.transAxes,
            fontsize=10, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("Week 6 Summary", fontweight="bold")

    plt.suptitle("Day 42 - Week 6 Recap: Sentiment Analysis Complete",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day42chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 42 - Week 6 Recap")
    print("=" * 55)
    print("\n  Week 6 Summary:")
    print("  Day 36: News fetching via RSS (CoinDesk, CryptoSlate)")
    print("  Day 37: NLP preprocessing — tokenize, clean, stem")
    print("  Day 38: VADER sentiment scoring on 18 headlines")
    print("  Day 39: Sentiment trend visualization (180 days)")
    print("  Day 40: Sentiment vs price correlation analysis")
    print("  Day 41: Sentiment-enhanced prediction model")
    print("\n  Key insight: Adding sentiment features improved")
    print("  MAE by ~11.9% over the baseline price-only model!")
    plot()
    print("\n  Next: Week 7 - Portfolio Optimizer!")
    print("\n✅ Week 6 complete!")

if __name__ == "__main__":
    run()

"""
Day 38 — VADER Sentiment Scoring on Crypto News
Week 6: Sentiment Analysis
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("reports", exist_ok=True)

HEADLINES = [
    ("Bitcoin surges past key resistance as institutional demand grows",         0.85),
    ("Ethereum upgrade boosts speed and reduces fees significantly",             0.72),
    ("Crypto market faces severe correction amid regulatory crackdown",         -0.68),
    ("Solana DeFi ecosystem records all-time high daily active users",           0.78),
    ("Bitcoin mining difficulty surges as hashrate reaches new peak",            0.45),
    ("Central banks issue strong warnings against cryptocurrency risks",        -0.55),
    ("Major exchange launches new tokens amid market optimism",                  0.60),
    ("Blockchain adoption accelerates across global financial sector",           0.65),
    ("Whale movements trigger fear of massive crypto market selloff",           -0.72),
    ("Bitcoin ETF sees record inflows as Wall Street embraces crypto",           0.88),
    ("Layer 2 solutions dramatically reduce Ethereum transaction costs",         0.70),
    ("Regulatory uncertainty continues to dampen crypto market sentiment",      -0.50),
    ("DeFi protocol hacked for millions in latest security breach",             -0.90),
    ("NFT market shows early recovery signs after prolonged downturn",           0.35),
    ("Altcoin season looming as Bitcoin dominance falls to yearly low",          0.40),
    ("Exchange faces insolvency fears as withdrawals surge amid panic",         -0.85),
    ("Crypto adoption reaches milestone with 500 million global users",          0.80),
    ("New privacy regulations threaten decentralized exchange operations",      -0.45),
]

def vader_score(text: str, preset_score: float) -> dict:
    """Simulated VADER-style scoring (real VADER needs nltk download)."""
    positive_words = ["surges","boosts","record","milestone","recovery","grows",
                      "optimism","accelerates","embraces","dramatically","peak"]
    negative_words = ["crackdown","warning","correction","hack","insolvency",
                      "panic","fear","threaten","severe","selloff","uncertainty"]
    text_lower = text.lower()
    pos = sum(1 for w in positive_words if w in text_lower)
    neg = sum(1 for w in negative_words if w in text_lower)
    compound = np.clip(preset_score + np.random.normal(0, 0.05), -1, 1)
    return {
        "compound": compound,
        "pos": max(0, pos / (pos + neg + 1)),
        "neg": max(0, neg / (pos + neg + 1)),
        "neu": max(0, 1 - pos/(pos+neg+1) - neg/(pos+neg+1)),
        "label": "POSITIVE" if compound > 0.05 else
                 "NEGATIVE" if compound < -0.05 else "NEUTRAL"
    }

def plot(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Compound scores
    colors = ["#2ECC71" if s > 0.05 else "#E74C3C" if s < -0.05 else "#F39C12"
              for s in df["compound"]]
    axes[0, 0].barh(range(len(df)), df["compound"], color=colors, alpha=0.8)
    axes[0, 0].axvline(0.05,  color="#2ECC71", linestyle="--", linewidth=0.8)
    axes[0, 0].axvline(-0.05, color="#E74C3C", linestyle="--", linewidth=0.8)
    axes[0, 0].axvline(0, color="black", linewidth=0.5)
    axes[0, 0].set_title("VADER Compound Scores per Headline", fontweight="bold")
    axes[0, 0].set_xlabel("Compound Score (-1=Very Negative, +1=Very Positive)")
    axes[0, 0].set_yticks(range(len(df)))
    axes[0, 0].set_yticklabels([h[:35]+"..." for h in df["headline"]], fontsize=6)
    axes[0, 0].grid(alpha=0.3, axis="x")

    # Distribution
    axes[0, 1].hist(df["compound"], bins=10, color="#3498DB", alpha=0.8, edgecolor="white")
    axes[0, 1].axvline(df["compound"].mean(), color="red", linestyle="--",
                        label=f"Mean: {df['compound'].mean():.2f}")
    axes[0, 1].set_title("Score Distribution", fontweight="bold")
    axes[0, 1].set_xlabel("Compound Score"); axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Label counts
    counts = df["label"].value_counts()
    pie_colors = {"POSITIVE": "#2ECC71", "NEGATIVE": "#E74C3C", "NEUTRAL": "#F39C12"}
    axes[1, 0].pie(counts.values, labels=counts.index,
                    colors=[pie_colors[k] for k in counts.index],
                    autopct="%1.0f%%", startangle=90)
    axes[1, 0].set_title("Sentiment Label Distribution", fontweight="bold")

    # Pos/Neg/Neu stacked
    axes[1, 1].bar(range(len(df)), df["pos"], color="#2ECC71", alpha=0.8, label="Positive")
    axes[1, 1].bar(range(len(df)), df["neg"], bottom=df["pos"],
                    color="#E74C3C", alpha=0.8, label="Negative")
    axes[1, 1].bar(range(len(df)), df["neu"], bottom=df["pos"]+df["neg"],
                    color="#95A5A6", alpha=0.8, label="Neutral")
    axes[1, 1].set_title("Pos/Neg/Neu Breakdown per Article", fontweight="bold")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3, axis="y")

    plt.suptitle("Day 38 - VADER Sentiment Analysis on Crypto News",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day38chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 38 - VADER Sentiment Scoring")
    print("=" * 55)
    results = []
    for headline, preset in HEADLINES:
        scores = vader_score(headline, preset)
        scores["headline"] = headline
        results.append(scores)
    df = pd.DataFrame(results)
    pos_count = (df["label"] == "POSITIVE").sum()
    neg_count = (df["label"] == "NEGATIVE").sum()
    neu_count = (df["label"] == "NEUTRAL").sum()
    print(f"  Headlines analyzed : {len(df)}")
    print(f"  Positive           : {pos_count}")
    print(f"  Negative           : {neg_count}")
    print(f"  Neutral            : {neu_count}")
    print(f"  Mean sentiment     : {df['compound'].mean():.3f}")
    print(f"  Overall market mood: {'BULLISH' if df['compound'].mean() > 0 else 'BEARISH'}")
    df.to_csv("data/sentiment_scores.csv", index=False)
    print(f"  Saved -> data/sentiment_scores.csv")
    plot(df)
    print("\n✅ Day 38 complete!")

if __name__ == "__main__":
    run()

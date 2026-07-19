"""
Day 37 — NLP Preprocessing: Tokenize, Clean, Stem
Week 6: Sentiment Analysis
"""
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from collections import Counter

os.makedirs("reports", exist_ok=True)

SAMPLE_HEADLINES = [
    "Bitcoin surges past key resistance level as institutional demand grows",
    "Ethereum network upgrade boosts transaction speed and reduces fees",
    "Crypto market faces correction amid regulatory uncertainty",
    "Solana DeFi ecosystem sees record daily active users",
    "Bitcoin mining difficulty hits all-time high as hashrate grows",
    "Central banks warn against cryptocurrency adoption risks",
    "Major exchange lists new altcoin tokens amid market optimism",
    "Blockchain technology adoption accelerates in financial sector",
    "Crypto whale movements signal potential market volatility ahead",
    "Bitcoin ETF sees record inflows as institutional interest surges",
    "Ethereum Layer 2 solutions reduce transaction costs significantly",
    "Regulatory framework for crypto assets expected by year end",
    "DeFi protocol launches new yield farming opportunities",
    "NFT market shows signs of recovery after prolonged downturn",
    "Altcoin season predicted as Bitcoin dominance weakens",
]

STOPWORDS = {
    "the","a","an","is","in","it","of","to","and","or","for","on","at",
    "by","as","be","are","was","with","has","have","had","that","this",
    "from","but","not","they","their","been","will","more","also","over",
    "when","than","into","after","amid","sees","new","key","also","past",
}

CRYPTO_STEMS = {
    "bitcoins": "bitcoin", "btcs": "btc", "ethereums": "ethereum",
    "cryptos": "crypto", "cryptocurrencies": "crypto", "cryptocurrency": "crypto",
    "blockchains": "blockchain", "exchanges": "exchange", "tokens": "token",
    "regulations": "regulation", "regulatory": "regulation",
    "institutional": "institution", "institutions": "institution",
}

def tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 3]
    tokens = [CRYPTO_STEMS.get(t, t) for t in tokens]
    return tokens

def plot(raw_freq, clean_freq, all_tokens):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    top_raw = pd.Series(raw_freq).nlargest(12)
    axes[0].barh(top_raw.index[::-1], top_raw.values[::-1], color="#E74C3C", alpha=0.8)
    axes[0].set_title("Top Words BEFORE Cleaning", fontweight="bold")
    axes[0].grid(alpha=0.3, axis="x")

    top_clean = pd.Series(clean_freq).nlargest(12)
    axes[1].barh(top_clean.index[::-1], top_clean.values[::-1], color="#2ECC71", alpha=0.8)
    axes[1].set_title("Top Words AFTER Cleaning", fontweight="bold")
    axes[1].grid(alpha=0.3, axis="x")

    lengths = [len(t) for t in all_tokens]
    axes[2].hist(lengths, bins=range(3, 15), color="#3498DB", alpha=0.8, edgecolor="white")
    axes[2].set_title("Token Length Distribution", fontweight="bold")
    axes[2].set_xlabel("Token Length"); axes[2].set_ylabel("Frequency")
    axes[2].grid(alpha=0.3, axis="y")

    plt.suptitle("Day 37 - NLP Preprocessing Pipeline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day37chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 37 - NLP Preprocessing")
    print("=" * 55)

    # Raw word frequency
    raw_words = []
    for h in SAMPLE_HEADLINES:
        raw_words.extend(h.lower().split())
    raw_freq = Counter(raw_words)

    # Clean tokens
    all_tokens = []
    processed  = []
    for h in SAMPLE_HEADLINES:
        tokens = tokenize(h)
        all_tokens.extend(tokens)
        processed.append({"original": h, "tokens": tokens, "n_tokens": len(tokens)})

    clean_freq = Counter(all_tokens)

    print(f"  Headlines processed  : {len(SAMPLE_HEADLINES)}")
    print(f"  Raw vocab size       : {len(raw_freq)}")
    print(f"  Clean vocab size     : {len(clean_freq)}")
    print(f"  Total clean tokens   : {len(all_tokens)}")
    print(f"\n  Sample tokenization:")
    for p in processed[:3]:
        print(f"    '{p['original'][:50]}...'")
        print(f"     -> {p['tokens']}")

    pd.DataFrame(processed).to_csv("data/nlp_tokens.csv", index=False)
    print(f"\n  Saved -> data/nlp_tokens.csv")
    plot(raw_freq, clean_freq, all_tokens)
    print("\n✅ Day 37 complete!")

if __name__ == "__main__":
    run()

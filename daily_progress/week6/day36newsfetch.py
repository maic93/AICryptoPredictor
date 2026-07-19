"""
Day 36 — Fetch Crypto News via RSS
Week 6: Sentiment Analysis
"""
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import os, re

os.makedirs("reports", exist_ok=True)
os.makedirs("data", exist_ok=True)

RSS_FEEDS = {
    "CoinDesk":      "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CryptoSlate":   "https://cryptoslate.com/feed/",
    "Bitcoin.com":   "https://news.bitcoin.com/feed/",
}

KEYWORDS = ["bitcoin","btc","ethereum","eth","crypto","blockchain",
            "solana","binance","defi","nft","altcoin"]

def fetch_rss(name: str, url: str) -> list:
    articles = []
    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(resp.content)
        for item in root.iter("item"):
            title = item.findtext("title") or ""
            desc  = item.findtext("description") or ""
            pub   = item.findtext("pubDate") or ""
            text  = (title + " " + desc).lower()
            if any(k in text for k in KEYWORDS):
                articles.append({"source": name, "title": title,
                                  "desc": desc[:200], "pubDate": pub})
    except Exception as e:
        print(f"  [{name}] Error: {e}")
    return articles

def simulate_articles() -> list:
    """Fallback simulated articles if RSS feeds are blocked."""
    headlines = [
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
    ]
    import random
    random.seed(42)
    articles = []
    for h in headlines:
        articles.append({"source": random.choice(list(RSS_FEEDS.keys())),
                          "title": h, "desc": h, "pubDate": datetime.utcnow().isoformat()})
    return articles

def plot(articles: list):
    if not articles:
        return
    sources = pd.Series([a["source"] for a in articles]).value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(sources.index, sources.values, color=["#F7931A","#627EEA","#2ECC71"][:len(sources)])
    axes[0].set_title("Articles per Source", fontweight="bold")
    axes[0].set_ylabel("Count"); axes[0].grid(alpha=0.3, axis="y")

    # Word frequency
    all_text = " ".join(a["title"].lower() for a in articles)
    words    = re.findall(r"\b[a-z]{4,}\b", all_text)
    stop     = {"that","this","with","from","have","will","been","they",
                "their","said","more","also","over","when","than","into"}
    freq     = pd.Series([w for w in words if w not in stop]).value_counts().head(15)
    axes[1].barh(freq.index[::-1], freq.values[::-1], color="#3498DB", alpha=0.8)
    axes[1].set_title("Top 15 Words in Headlines", fontweight="bold")
    axes[1].grid(alpha=0.3, axis="x")

    plt.suptitle("Day 36 - Crypto News Fetch & Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day36chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 36 - Fetch Crypto News")
    print("=" * 55)
    all_articles = []
    for name, url in RSS_FEEDS.items():
        arts = fetch_rss(name, url)
        all_articles.extend(arts)
        print(f"  [{name}] {len(arts)} articles fetched")

    if not all_articles:
        print("  RSS feeds blocked on runner — using simulated articles")
        all_articles = simulate_articles()

    print(f"\n  Total articles : {len(all_articles)}")
    print("\n  Sample headlines:")
    for a in all_articles[:5]:
        print(f"    [{a['source']}] {a['title'][:70]}...")

    pd.DataFrame(all_articles).to_csv("data/news_articles.csv", index=False)
    print(f"\n  Saved -> data/news_articles.csv")
    plot(all_articles)
    print("\n✅ Day 36 complete!")

if __name__ == "__main__":
    run()

"""
Day 55 — Portfolio Pie Chart Component
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

COINS    = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"]
LABELS   = ["BTC","ETH","SOL","BNB"]
COLORS   = ["#F7931A","#627EEA","#9945FF","#F0B90B"]
PORTFOLIOS = {
    "Equal Weight":   np.array([0.25, 0.25, 0.25, 0.25]),
    "BTC Heavy":      np.array([0.60, 0.20, 0.10, 0.10]),
    "Sharpe-Optimal": np.array([0.35, 0.30, 0.25, 0.10]),
    "Risk-Parity":    np.array([0.20, 0.25, 0.30, 0.25]),
}

def fetch_prices():
    prices = {}
    for coin in COINS:
        df = yf.download(coin, period="1mo", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        close = df["Close"].squeeze().dropna()
        prices[coin.replace("-USD","")] = float(close.iloc[-1].item())
    return prices

def compute_portfolio(weights: np.ndarray, prices: dict,
                       capital: float = 10_000) -> dict:
    holdings = {}
    for label, w in zip(LABELS, weights):
        value  = capital * w
        amount = value / prices[label]
        holdings[label] = {"weight": w, "value": value,
                            "amount": amount, "price": prices[label]}
    return holdings

def plot(prices: dict):
    fig = plt.figure(figsize=(16, 10), facecolor="#0e1117")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    for idx, (port_name, weights) in enumerate(PORTFOLIOS.items()):
        row = idx // 4
        col = idx % 4
        ax  = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#1c1c2e")

        holdings = compute_portfolio(weights, prices)
        values   = [h["value"] for h in holdings.values()]

        wedges, texts, autotexts = ax.pie(
            values, labels=LABELS, colors=COLORS,
            autopct="%1.0f%%", startangle=90,
            wedgeprops={"edgecolor":"#0e1117","linewidth":2},
            textprops={"color":"white","fontsize":9},
        )
        for at in autotexts:
            at.set_color("white"); at.set_fontsize(8)

        ax.set_title(port_name, color="white", fontweight="bold", fontsize=10)
        # Holdings table below
        for i, (label, h) in enumerate(holdings.items()):
            ax.text(-1.5, 0.9 - i*0.35,
                     f"{label}: {h['amount']:.4f} @ ${h['price']:,.0f}",
                     fontsize=7, color="#aaa", transform=ax.transAxes)

    # Price ticker strip
    ax_strip = fig.add_subplot(gs[1, :])
    ax_strip.set_facecolor("#262730")
    ax_strip.axis("off")
    ticker_str = "   |   ".join(
        [f"{label}: ${prices[label]:,.2f}" for label in LABELS]
    )
    ax_strip.text(0.5, 0.6, ticker_str, ha="center", va="center",
                   color="#F7931A", fontsize=14, fontweight="bold",
                   transform=ax_strip.transAxes)
    ax_strip.text(0.5, 0.2,
                   f"Capital: $10,000 | Updated: {datetime.utcnow().strftime('%H:%M UTC')}",
                   ha="center", color="#aaa", fontsize=10, transform=ax_strip.transAxes)

    fig.suptitle("Day 55 - Portfolio Pie Chart Component",
                  fontsize=13, fontweight="bold", color="white")
    out = "reports/day55chart.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 55 - Portfolio Pie Chart Component")
    print("=" * 55)
    prices = fetch_prices()
    print(f"\n  Live Prices:")
    for label, price in prices.items():
        print(f"    {label}: ${price:,.2f}")
    print(f"\n  Portfolio Allocations ($10,000 capital):")
    for port_name, weights in PORTFOLIOS.items():
        holdings = compute_portfolio(weights, prices)
        print(f"\n  [{port_name}]")
        for label, h in holdings.items():
            print(f"    {label}: {h['amount']:.4f} coins = ${h['value']:,.2f}")
    plot(prices)
    print("\n  Not financial advice!")
    print("\n✅ Day 55 complete!")

if __name__ == "__main__":
    run()

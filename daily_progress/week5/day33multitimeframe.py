"""
Day 33 — Multi-Timeframe Signal Analysis
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

TIMEFRAMES = {"1d": ("2y", 1), "1wk": ("5y", 7), "1mo": ("10y", 30)}

def get_signal(ticker="BTC-USD", interval="1d", period="2y") -> dict:
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
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

    trend = "BULLISH" if close.iloc[-1] > close.rolling(50).mean().iloc[-1] else "BEARISH"
    signal = "BUY"  if rsi < 45 and macd > sig else \
             "SELL" if rsi > 55 and macd < sig else "HOLD"

    return {"interval": interval, "rsi": rsi, "macd": macd,
            "signal": signal, "trend": trend, "close": close}

def plot(signals: dict):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    colors_map = {"BUY": "#2ECC71", "SELL": "#E74C3C", "HOLD": "#F39C12"}
    trend_map  = {"BULLISH": "#2ECC71", "BEARISH": "#E74C3C"}

    for col, (tf, data) in enumerate(signals.items()):
        # Price chart
        close = data["close"]
        axes[0, col].plot(close.index, close.values,
                           color="#F7931A", linewidth=1.2)
        ma50 = close.rolling(50).mean()
        axes[0, col].plot(close.index, ma50.values,
                           color="#3498DB", linewidth=1, linestyle="--", label="50 MA")
        axes[0, col].set_title(f"BTC {tf} | {data['trend']}",
                                fontweight="bold",
                                color=trend_map[data["trend"]])
        axes[0, col].legend(fontsize=7); axes[0, col].grid(alpha=0.3)
        axes[0, col].tick_params(axis="x", rotation=30, labelsize=7)

        # Signal summary
        ax = axes[1, col]
        ax.axis("off")
        sig_color = colors_map[data["signal"]]
        ax.text(0.5, 0.7, data["signal"], ha="center", va="center",
                fontsize=28, fontweight="bold", color=sig_color,
                transform=ax.transAxes)
        ax.text(0.5, 0.4, f"RSI: {data['rsi']:.1f}", ha="center",
                fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.2, f"MACD: {data['macd']:.2f}", ha="center",
                fontsize=12, transform=ax.transAxes)
        ax.set_facecolor("#f8f9fa")

    plt.suptitle("Day 33 - Multi-Timeframe Signal Analysis (BTC)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day33chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 33 - Multi-Timeframe Signal Analysis")
    print("=" * 55)
    signals = {}
    for tf, (period, _) in TIMEFRAMES.items():
        print(f"\n  Fetching {tf} data...")
        try:
            data = get_signal("BTC-USD", tf, period)
            signals[tf] = data
            print(f"  RSI: {data['rsi']:.1f} | MACD: {data['macd']:.4f} | "
                  f"Signal: {data['signal']} | Trend: {data['trend']}")
        except Exception as e:
            print(f"  Error for {tf}: {e}")

    if signals:
        plot(signals)
        all_signals = [v["signal"] for v in signals.values()]
        buys  = all_signals.count("BUY")
        sells = all_signals.count("SELL")
        consensus = "STRONG BUY" if buys >= 2 else \
                    "STRONG SELL" if sells >= 2 else "MIXED"
        print(f"\n  Multi-timeframe consensus: {consensus}")
    print("\n✅ Day 33 complete!")

if __name__ == "__main__":
    run()

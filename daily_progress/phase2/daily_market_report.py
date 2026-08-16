"""
daily_market_report.py
Runs every day via GitHub Actions.
Fetches live crypto data and writes a rich markdown market report.
No images — pure markdown with tables, emoji indicators and real data.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
from datetime import datetime, date, timedelta

os.makedirs("market_reports", exist_ok=True)
os.makedirs("data", exist_ok=True)

COINS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "BNB-USD": "Binance Coin",
    "ADA-USD": "Cardano",
    "DOT-USD": "Polkadot",
}

SIGNAL_LOG = "data/signal_log.json"
PERF_LOG   = "data/performance_log.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def arrow(val: float) -> str:
    return "▲" if val > 0 else "▼" if val < 0 else "→"

def sentiment_emoji(score: float) -> str:
    if score >  0.5: return "🟢 Very Bullish"
    if score >  0.1: return "🟡 Bullish"
    if score > -0.1: return "⚪ Neutral"
    if score > -0.5: return "🟠 Bearish"
    return "🔴 Very Bearish"

def rsi_label(rsi: float) -> str:
    if rsi > 70: return "Overbought"
    if rsi < 30: return "Oversold"
    return "Neutral"

def fear_greed_label(val: int) -> str:
    if val < 25: return "Extreme Fear"
    if val < 45: return "Fear"
    if val < 55: return "Neutral"
    if val < 75: return "Greed"
    return "Extreme Greed"

# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_coin(ticker: str) -> dict:
    df = yf.download(ticker, period="3mo", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.dropna(inplace=True)
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    price   = float(close.iloc[-1].item())
    open_   = float(df["Open"].squeeze().iloc[-1].item())
    high    = float(df["High"].squeeze().iloc[-1].item())
    low     = float(df["Low"].squeeze().iloc[-1].item())
    vol     = float(volume.iloc[-1].item())
    vol_avg = float(volume.rolling(20).mean().iloc[-1].item())

    ret1d  = float((close.iloc[-1] - close.iloc[-2])  / close.iloc[-2]  * 100)
    ret7d  = float((close.iloc[-1] - close.iloc[-7])  / close.iloc[-7]  * 100)
    ret30d = float((close.iloc[-1] - close.iloc[-30]) / close.iloc[-30] * 100)

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = float((100 - 100 / (1 + gain / loss)).iloc[-1].item())

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = float((ema12 - ema26).iloc[-1].item())
    macd_sig = float((ema12 - ema26).ewm(span=9).mean().iloc[-1].item())

    # Bollinger Bands
    sma20 = float(close.rolling(20).mean().iloc[-1].item())
    std20 = float(close.rolling(20).std().iloc[-1].item())
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_pct   = (price - bb_lower) / (bb_upper - bb_lower) * 100

    # Volatility
    vol_30d = float(close.pct_change().rolling(30).std().iloc[-1].item() * np.sqrt(252) * 100)

    # Signal
    score = np.clip(
        (50 - rsi) * -0.4 +
        (np.sign(macd - macd_sig) * 20) +
        (np.sign(price - sma20) * 15),
        -100, 100
    )
    signal = "BUY" if score > 20 else "SELL" if score < -20 else "HOLD"

    return {
        "ticker": ticker, "price": price, "open": open_,
        "high": high, "low": low,
        "ret1d": ret1d, "ret7d": ret7d, "ret30d": ret30d,
        "rsi": rsi, "macd": macd, "macd_sig": macd_sig,
        "sma20": sma20, "bb_upper": bb_upper, "bb_lower": bb_lower,
        "bb_pct": bb_pct, "vol_30d": vol_30d,
        "volume": vol, "vol_avg": vol_avg,
        "score": float(score), "signal": signal,
        "above_sma20": price > sma20,
    }

def compute_fear_greed(coins: list) -> int:
    """Simple Fear & Greed approximation from RSI + momentum."""
    rsi_avg  = np.mean([c["rsi"] for c in coins])
    mom_avg  = np.mean([c["ret7d"] for c in coins])
    raw      = (rsi_avg - 50) * 0.6 + np.clip(mom_avg * 2, -30, 30)
    return int(np.clip(50 + raw, 0, 100))

def simulate_sentiment(coins: list) -> float:
    """Derive a sentiment proxy from price action."""
    bullish  = sum(1 for c in coins if c["ret7d"] > 2)
    bearish  = sum(1 for c in coins if c["ret7d"] < -2)
    return np.clip((bullish - bearish) / len(coins), -1, 1)

# ── Report generation ─────────────────────────────────────────────────────────

def build_report(coins: list, day_num: int) -> str:
    today      = date.today()
    ts         = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    fg_index   = compute_fear_greed(coins)
    fg_label   = fear_greed_label(fg_index)
    sentiment  = simulate_sentiment(coins)
    sent_label = sentiment_emoji(sentiment)

    buys  = [c for c in coins if c["signal"] == "BUY"]
    sells = [c for c in coins if c["signal"] == "SELL"]
    holds = [c for c in coins if c["signal"] == "HOLD"]

    btc = next(c for c in coins if c["ticker"] == "BTC-USD")

    lines = []
    lines.append(f"# Daily Crypto Market Report — {today}")
    lines.append(f"")
    lines.append(f"> **Day {day_num}** of 30-day live market tracking | Generated: {ts}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # Market overview
    lines.append(f"## Market Overview")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Fear & Greed Index | **{fg_index}/100** — {fg_label} |")
    lines.append(f"| Market Sentiment   | {sent_label} |")
    lines.append(f"| BTC Price          | **${btc['price']:,.2f}** ({arrow(btc['ret1d'])} {btc['ret1d']:+.2f}% 24h) |")
    lines.append(f"| BTC Dominance Proxy| {btc['ret7d']:+.1f}% vs alts avg: {np.mean([c['ret7d'] for c in coins if c['ticker']!='BTC-USD']):+.1f}% |")
    lines.append(f"| Active Signals     | {len(buys)} BUY · {len(sells)} SELL · {len(holds)} HOLD |")
    lines.append(f"")

    # Price table
    lines.append(f"## Price Dashboard")
    lines.append(f"")
    lines.append(f"| Coin | Price | 24h | 7d | 30d | RSI | Signal | Score |")
    lines.append(f"|------|-------|-----|----|-----|-----|--------|-------|")
    for c in coins:
        sig_icon = "🟢" if c["signal"]=="BUY" else "🔴" if c["signal"]=="SELL" else "⚪"
        lines.append(
            f"| **{c['ticker'].replace('-USD','')}** "
            f"| ${c['price']:,.2f} "
            f"| {arrow(c['ret1d'])} {c['ret1d']:+.2f}% "
            f"| {arrow(c['ret7d'])} {c['ret7d']:+.2f}% "
            f"| {arrow(c['ret30d'])} {c['ret30d']:+.2f}% "
            f"| {c['rsi']:.0f} ({rsi_label(c['rsi'])}) "
            f"| {sig_icon} {c['signal']} "
            f"| {c['score']:+.0f} |"
        )
    lines.append(f"")

    # Technical analysis
    lines.append(f"## Technical Analysis")
    lines.append(f"")
    lines.append(f"| Coin | SMA20 | BB% | MACD | Vol 30d | Vol vs Avg |")
    lines.append(f"|------|-------|-----|------|---------|------------|")
    for c in coins:
        above = "above" if c["above_sma20"] else "below"
        vol_ratio = c["volume"] / c["vol_avg"] if c["vol_avg"] > 0 else 1
        lines.append(
            f"| **{c['ticker'].replace('-USD','')}** "
            f"| ${c['sma20']:,.0f} ({above}) "
            f"| {c['bb_pct']:.0f}% "
            f"| {'+' if c['macd'] > c['macd_sig'] else '-'} "
            f"| {c['vol_30d']:.0f}% "
            f"| {vol_ratio:.1f}x |"
        )
    lines.append(f"")

    # OHLC
    lines.append(f"## Today's OHLC")
    lines.append(f"")
    lines.append(f"| Coin | Open | High | Low | Close | Range |")
    lines.append(f"|------|------|------|-----|-------|-------|")
    for c in coins:
        rng = c["high"] - c["low"]
        lines.append(
            f"| **{c['ticker'].replace('-USD','')}** "
            f"| ${c['open']:,.2f} "
            f"| ${c['high']:,.2f} "
            f"| ${c['low']:,.2f} "
            f"| ${c['price']:,.2f} "
            f"| ${rng:,.2f} |"
        )
    lines.append(f"")

    # Signal summary
    lines.append(f"## Signal Summary")
    lines.append(f"")
    if buys:
        lines.append(f"### 🟢 BUY Signals")
        for c in buys:
            lines.append(f"- **{c['ticker'].replace('-USD','')}** @ ${c['price']:,.2f} "
                          f"— RSI: {c['rsi']:.0f}, Score: {c['score']:+.0f}, "
                          f"7d: {c['ret7d']:+.2f}%")
        lines.append(f"")
    if sells:
        lines.append(f"### 🔴 SELL Signals")
        for c in sells:
            lines.append(f"- **{c['ticker'].replace('-USD','')}** @ ${c['price']:,.2f} "
                          f"— RSI: {c['rsi']:.0f}, Score: {c['score']:+.0f}, "
                          f"7d: {c['ret7d']:+.2f}%")
        lines.append(f"")
    if holds:
        lines.append(f"### ⚪ HOLD")
        for c in holds:
            lines.append(f"- **{c['ticker'].replace('-USD','')}** @ ${c['price']:,.2f} "
                          f"— RSI: {c['rsi']:.0f}, Score: {c['score']:+.0f}")
        lines.append(f"")

    # Market narrative
    lines.append(f"## Market Narrative")
    lines.append(f"")
    if fg_index >= 60:
        mood = "Greed is driving the market. Momentum is strong but caution is advised near resistance levels."
    elif fg_index <= 40:
        mood = "Fear is dominating. Historically, extreme fear has presented accumulation opportunities."
    else:
        mood = "The market is in a neutral zone, consolidating before the next directional move."
    lines.append(f"{mood}")
    lines.append(f"")
    lines.append(f"BTC is trading **{('above' if btc['above_sma20'] else 'below')} its 20-day SMA** "
                  f"at ${btc['sma20']:,.0f}, with RSI at {btc['rsi']:.0f} indicating "
                  f"**{rsi_label(btc['rsi'])}** conditions. "
                  f"The Bollinger Band position at {btc['bb_pct']:.0f}% suggests "
                  f"{'price is extended toward the upper band.' if btc['bb_pct'] > 70 else 'price is compressed near the lower band.' if btc['bb_pct'] < 30 else 'price is within normal range.'}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"*Data sourced from Yahoo Finance. Educational purposes only — not financial advice.*")
    lines.append(f"*Part of [AICryptoPredictor](https://github.com/maic93/AICryptoPredictor) — 30-day live market tracking.*")

    return "\n".join(lines)

# ── Logging ───────────────────────────────────────────────────────────────────

def update_signal_log(coins: list):
    log = []
    if os.path.exists(SIGNAL_LOG):
        with open(SIGNAL_LOG) as f:
            log = json.load(f)
    entry = {
        "date": date.today().isoformat(),
        "signals": {c["ticker"].replace("-USD",""): {
            "signal": c["signal"],
            "score":  c["score"],
            "price":  c["price"],
            "rsi":    c["rsi"],
        } for c in coins}
    }
    log.append(entry)
    log = log[-60:]  # keep last 60 days
    with open(SIGNAL_LOG, "w") as f:
        json.dump(log, f, indent=2)

def update_performance_log(coins: list):
    log = []
    if os.path.exists(PERF_LOG):
        with open(PERF_LOG) as f:
            log = json.load(f)
    entry = {
        "date":       date.today().isoformat(),
        "btc_price":  next(c["price"] for c in coins if c["ticker"]=="BTC-USD"),
        "eth_price":  next(c["price"] for c in coins if c["ticker"]=="ETH-USD"),
        "avg_ret7d":  float(np.mean([c["ret7d"]  for c in coins])),
        "avg_ret30d": float(np.mean([c["ret30d"] for c in coins])),
        "avg_rsi":    float(np.mean([c["rsi"]    for c in coins])),
        "n_buy":      sum(1 for c in coins if c["signal"]=="BUY"),
        "n_sell":     sum(1 for c in coins if c["signal"]=="SELL"),
        "n_hold":     sum(1 for c in coins if c["signal"]=="HOLD"),
    }
    log.append(entry)
    log = log[-60:]
    with open(PERF_LOG, "w") as f:
        json.dump(log, f, indent=2)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from datetime import date as d
    PHASE2_START = d(2026, 8, 16)  # UPDATE THIS to day after day 56
    day_num = max(1, min((d.today() - PHASE2_START).days + 1, 30))

    print(f"Daily Market Report — Day {day_num}/30")
    print(f"Date: {date.today()} | {datetime.utcnow().strftime('%H:%M UTC')}")
    print("=" * 55)

    coins = []
    for ticker, name in COINS.items():
        try:
            data = fetch_coin(ticker)
            coins.append(data)
            sig_icon = "BUY " if data["signal"]=="BUY" else "SELL" if data["signal"]=="SELL" else "HOLD"
            print(f"  {ticker:<12} ${data['price']:>10,.2f} | "
                  f"RSI:{data['rsi']:>5.1f} | {sig_icon} ({data['score']:+.0f})")
        except Exception as e:
            print(f"  {ticker:<12} ERROR: {e}")

    if not coins:
        print("ERROR: No coin data fetched!")
        return

    # Build and save report
    report = build_report(coins, day_num)
    report_path = f"market_reports/{date.today()}.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved -> {report_path}")

    # Update logs
    update_signal_log(coins)
    update_performance_log(coins)
    print(f"  Logs updated  -> {SIGNAL_LOG}")
    print(f"                -> {PERF_LOG}")

    # Print summary
    btc = next(c for c in coins if c["ticker"]=="BTC-USD")
    fg  = compute_fear_greed(coins)
    print(f"\n  Fear & Greed : {fg}/100 ({fear_greed_label(fg)})")
    print(f"  BTC          : ${btc['price']:,.2f} ({btc['ret1d']:+.2f}% 24h)")
    print(f"  Signals      : {sum(1 for c in coins if c['signal']=='BUY')} BUY / "
          f"{sum(1 for c in coins if c['signal']=='SELL')} SELL / "
          f"{sum(1 for c in coins if c['signal']=='HOLD')} HOLD")
    print(f"\nDay {day_num}/30 complete!")

if __name__ == "__main__":
    main()

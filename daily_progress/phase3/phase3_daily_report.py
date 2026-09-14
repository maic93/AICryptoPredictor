"""
phase3_daily_report.py
Phase 3 — Enhanced Daily Market Report (90 days)
Runs every day via GitHub Actions.

Daily: Live prices + signals + Fear&Greed + model retraining + performance tracking
Sunday: Full weekly deep dive added to report

No images — pure markdown, zero repo bloat.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import json
import os
from datetime import datetime, date, timedelta
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("market_reports", exist_ok=True)
os.makedirs("data", exist_ok=True)

PHASE3_START = date(2026, 9, 15)  # UPDATE to your actual Phase 3 start date

COINS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "BNB-USD": "Binance Coin",
    "ADA-USD": "Cardano",
    "DOT-USD": "Polkadot",
}

SIGNAL_LOG      = "data/signal_log.json"
PERF_LOG        = "data/performance_log.json"
MODEL_PERF_LOG  = "data/model_performance.json"
WEEKLY_LOG      = "data/weekly_reviews.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def arrow(v):   return "▲" if v > 0 else "▼" if v < 0 else "→"
def rsi_label(r): return "Overbought" if r > 70 else "Oversold" if r < 30 else "Neutral"
def fg_label(v):
    if v < 25: return "Extreme Fear"
    if v < 45: return "Fear"
    if v < 55: return "Neutral"
    if v < 75: return "Greed"
    return "Extreme Greed"

def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default

def save_json(path, data, max_entries=120):
    if isinstance(data, list):
        data = data[-max_entries:]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ── Fear & Greed Index ────────────────────────────────────────────────────────

def fetch_fear_greed() -> dict:
    """Fetch real Fear & Greed index from alternative.me (free, no API key)."""
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=8
        )
        data = resp.json()["data"][0]
        return {
            "value": int(data["value"]),
            "label": data["value_classification"],
            "source": "alternative.me"
        }
    except Exception:
        # Fallback: compute from price action
        return None

def compute_fg_fallback(coins: list) -> dict:
    rsi_avg = np.mean([c["rsi"] for c in coins])
    mom_avg = np.mean([c["ret7d"] for c in coins])
    raw     = (rsi_avg - 50) * 0.6 + np.clip(mom_avg * 2, -30, 30)
    value   = int(np.clip(50 + raw, 0, 100))
    return {"value": value, "label": fg_label(value), "source": "computed"}

# ── Coin data ─────────────────────────────────────────────────────────────────

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

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = float((100 - 100 / (1 + gain / loss)).iloc[-1].item())

    ema12    = close.ewm(span=12).mean()
    ema26    = close.ewm(span=26).mean()
    macd     = float((ema12 - ema26).iloc[-1].item())
    macd_sig = float((ema12 - ema26).ewm(span=9).mean().iloc[-1].item())

    sma20    = float(close.rolling(20).mean().iloc[-1].item())
    std20    = float(close.rolling(20).std().iloc[-1].item())
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_pct   = (price - bb_lower) / (bb_upper - bb_lower) * 100

    vol_30d = float(close.pct_change().rolling(30).std().iloc[-1].item() * np.sqrt(252) * 100)

    score = float(np.clip(
        (50 - rsi) * -0.4 +
        np.sign(macd - macd_sig) * 20 +
        np.sign(price - sma20) * 15,
        -100, 100
    ))
    signal = "BUY" if score > 20 else "SELL" if score < -20 else "HOLD"

    return {
        "ticker": ticker, "price": price, "open": open_,
        "high": high, "low": low, "volume": vol, "vol_avg": vol_avg,
        "ret1d": ret1d, "ret7d": ret7d, "ret30d": ret30d,
        "rsi": rsi, "macd": macd, "macd_sig": macd_sig,
        "sma20": sma20, "bb_upper": bb_upper, "bb_lower": bb_lower,
        "bb_pct": bb_pct, "vol_30d": vol_30d,
        "score": score, "signal": signal,
        "above_sma20": price > sma20,
        "close_series": close,
    }

# ── Model retraining & performance tracking ───────────────────────────────────

def retrain_and_evaluate(ticker="BTC-USD") -> dict:
    """Retrain XGBoost on latest 2y data and log MAE/R²."""
    try:
        df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        df.dropna(inplace=True)
        close = df["Close"].squeeze()
        for lag in [1, 2, 3, 5, 7, 14]:
            df[f"lag{lag}"] = close.shift(lag)
        for w in [7, 14, 30]:
            df[f"rollMean{w}"] = close.rolling(w).mean()
            df[f"rollStd{w}"]  = close.rolling(w).std()
        df["return1d"] = close.pct_change()
        df["return7d"]  = close.pct_change(7)
        df["target"]   = close.shift(-1)
        df.dropna(inplace=True)
        feat_cols = [str(c) for c in df.columns
                     if c not in ["target","Open","High","Low","Close","Volume"]]
        df.columns = [str(c) for c in df.columns]
        X = df[feat_cols]; y = df["target"]
        split = int(len(X) * 0.8)
        model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        model.fit(X.iloc[:split], y.iloc[:split])
        preds  = model.predict(X.iloc[split:])
        actual = y.iloc[split:].values
        n      = min(len(preds), len(actual))
        mae    = float(mean_absolute_error(actual[:n], preds[:n]))
        r2     = float(r2_score(actual[:n], preds[:n]))
        next_p = float(model.predict(X.iloc[[-1]])[0])
        return {"mae": mae, "r2": r2, "next_pred": next_p,
                "current_price": float(close.iloc[-1].item()),
                "status": "ok"}
    except Exception as e:
        return {"mae": None, "r2": None, "next_pred": None,
                "current_price": None, "status": f"error: {e}"}

def update_model_log(model_result: dict, day_num: int):
    log = load_json(MODEL_PERF_LOG, [])
    entry = {
        "date":    date.today().isoformat(),
        "day":     day_num,
        "mae":     model_result["mae"],
        "r2":      model_result["r2"],
        "next_pred": model_result["next_pred"],
        "price":   model_result["current_price"],
    }
    log.append(entry)
    save_json(MODEL_PERF_LOG, log)

    # Drift detection: compare to 7-day average MAE
    if len(log) >= 8 and log[-1]["mae"] and log[-8]["mae"]:
        recent_avg = np.mean([e["mae"] for e in log[-7:-1] if e["mae"]])
        current    = log[-1]["mae"]
        drift_pct  = (current - recent_avg) / recent_avg * 100
        return drift_pct
    return 0.0

# ── Weekly deep dive ──────────────────────────────────────────────────────────

def weekly_deep_dive(coins: list, day_num: int) -> str:
    """Generate a full weekly review section (runs every Sunday)."""
    sig_log  = load_json(SIGNAL_LOG, [])
    mod_log  = load_json(MODEL_PERF_LOG, [])
    perf_log = load_json(PERF_LOG, [])

    lines = []
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## 📅 Weekly Deep Dive — Week {(day_num - 1) // 7 + 1}")
    lines.append(f"")

    # Best & worst performer this week
    sorted_coins = sorted(coins, key=lambda c: c["ret7d"], reverse=True)
    best  = sorted_coins[0]
    worst = sorted_coins[-1]
    lines.append(f"### Performance This Week")
    lines.append(f"")
    lines.append(f"| | Coin | 7-Day Return |")
    lines.append(f"|--|------|-------------|")
    lines.append(f"| Best  | **{best['ticker'].replace('-USD','')}** | {arrow(best['ret7d'])} {best['ret7d']:+.2f}% |")
    lines.append(f"| Worst | **{worst['ticker'].replace('-USD','')}** | {arrow(worst['ret7d'])} {worst['ret7d']:+.2f}% |")
    lines.append(f"")

    # Signal accuracy (were last week's signals correct?)
    if len(sig_log) >= 7:
        last_week = sig_log[-7:]
        correct = 0
        total   = 0
        for entry in last_week:
            for coin_label, data in entry.get("signals", {}).items():
                sig   = data.get("signal")
                ret1d = data.get("ret1d", 0)
                if sig == "BUY"  and ret1d > 0: correct += 1
                if sig == "SELL" and ret1d < 0: correct += 1
                if sig != "HOLD": total += 1
        accuracy = correct / total * 100 if total > 0 else 0
        lines.append(f"### Signal Accuracy (Last 7 Days)")
        lines.append(f"")
        lines.append(f"- Directional signals evaluated: **{total}**")
        lines.append(f"- Correct direction: **{correct}** ({accuracy:.1f}%)")
        lines.append(f"- {'Good accuracy — signals are working well!' if accuracy > 55 else 'Mixed results — market is choppy.'}")
        lines.append(f"")

    # Model performance trend
    if len(mod_log) >= 7:
        week_maes = [e["mae"] for e in mod_log[-7:] if e["mae"]]
        if week_maes:
            lines.append(f"### Model Performance This Week (XGBoost BTC)")
            lines.append(f"")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Best MAE  | ${min(week_maes):,.2f} |")
            lines.append(f"| Worst MAE | ${max(week_maes):,.2f} |")
            lines.append(f"| Avg MAE   | ${np.mean(week_maes):,.2f} |")
            trend = "improving" if week_maes[-1] < week_maes[0] else "degrading"
            lines.append(f"| Trend     | Model accuracy is **{trend}** this week |")
            lines.append(f"")

    # Hypothetical portfolio simulation
    lines.append(f"### Hypothetical Portfolio (Equal Weight, $10,000)")
    lines.append(f"")
    lines.append(f"| Coin | 7d Return | $10k → |")
    lines.append(f"|------|-----------|--------|")
    total_ret = 0
    for c in sorted_coins:
        val = 10_000 / len(coins) * (1 + c["ret7d"] / 100)
        total_ret += val
        lines.append(f"| {c['ticker'].replace('-USD','')} | {c['ret7d']:+.2f}% | ${val:,.2f} |")
    lines.append(f"| **Total** | | **${total_ret:,.2f}** |")
    lines.append(f"")
    ret_pct = (total_ret - 10_000) / 10_000 * 100
    lines.append(f"Equal-weight portfolio this week: **{ret_pct:+.2f}%**")
    lines.append(f"")

    # Save weekly summary
    weekly = load_json(WEEKLY_LOG, [])
    weekly.append({
        "date":        date.today().isoformat(),
        "week":        (day_num - 1) // 7 + 1,
        "best_coin":   best["ticker"],
        "worst_coin":  worst["ticker"],
        "portfolio_ret": ret_pct,
    })
    save_json(WEEKLY_LOG, weekly)

    return "\n".join(lines)

# ── Report builder ────────────────────────────────────────────────────────────

def build_report(coins: list, fg: dict, model: dict,
                 drift_pct: float, day_num: int) -> str:
    today  = date.today()
    ts     = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    is_sun = today.weekday() == 6

    buys  = [c for c in coins if c["signal"] == "BUY"]
    sells = [c for c in coins if c["signal"] == "SELL"]
    holds = [c for c in coins if c["signal"] == "HOLD"]
    btc   = next(c for c in coins if c["ticker"] == "BTC-USD")

    lines = []
    lines.append(f"# Daily Crypto Market Report — {today}")
    lines.append(f"")
    lines.append(f"> **Phase 3 — Day {day_num}/90** | Generated: {ts}")
    if is_sun:
        lines.append(f"> 📅 Sunday — Weekly Deep Dive included below")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # Market overview
    lines.append(f"## Market Overview")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Fear & Greed Index | **{fg['value']}/100** — {fg['label']} *(source: {fg['source']})* |")
    lines.append(f"| BTC Price | **${btc['price']:,.2f}** ({arrow(btc['ret1d'])} {btc['ret1d']:+.2f}% 24h) |")
    lines.append(f"| BTC vs SMA20 | {'Above' if btc['above_sma20'] else 'Below'} ${btc['sma20']:,.0f} |")
    lines.append(f"| Active Signals | {len(buys)} BUY · {len(sells)} SELL · {len(holds)} HOLD |")
    if model["status"] == "ok" and model["next_pred"]:
        pred_chg = (model["next_pred"] - btc["price"]) / btc["price"] * 100
        lines.append(f"| BTC Tomorrow (AI) | ${model['next_pred']:,.2f} ({pred_chg:+.2f}%) |")
        lines.append(f"| Model MAE | ${model['mae']:,.2f} | ")
        if abs(drift_pct) > 10:
            lines.append(f"| Model Drift | {'⚠️ Degrading' if drift_pct > 0 else '✅ Improving'} ({drift_pct:+.1f}% vs 7d avg) |")
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
    lines.append(f"| Coin | vs SMA20 | BB% | MACD | Vol 30d | Vol vs Avg |")
    lines.append(f"|------|----------|-----|------|---------|------------|")
    for c in coins:
        above     = "above" if c["above_sma20"] else "below"
        vol_ratio = c["volume"] / c["vol_avg"] if c["vol_avg"] > 0 else 1
        lines.append(
            f"| **{c['ticker'].replace('-USD','')}** "
            f"| {above} ${c['sma20']:,.0f} "
            f"| {c['bb_pct']:.0f}% "
            f"| {'+' if c['macd'] > c['macd_sig'] else '-'} "
            f"| {c['vol_30d']:.0f}% "
            f"| {vol_ratio:.1f}x |"
        )
    lines.append(f"")

    # OHLC
    lines.append(f"## OHLC Table")
    lines.append(f"")
    lines.append(f"| Coin | Open | High | Low | Close | Daily Range |")
    lines.append(f"|------|------|------|-----|-------|-------------|")
    for c in coins:
        lines.append(
            f"| **{c['ticker'].replace('-USD','')}** "
            f"| ${c['open']:,.2f} "
            f"| ${c['high']:,.2f} "
            f"| ${c['low']:,.2f} "
            f"| ${c['price']:,.2f} "
            f"| ${c['high']-c['low']:,.2f} |"
        )
    lines.append(f"")

    # Signals
    lines.append(f"## Signal Summary")
    lines.append(f"")
    if buys:
        lines.append(f"### 🟢 BUY")
        for c in buys:
            lines.append(f"- **{c['ticker'].replace('-USD','')}** @ ${c['price']:,.2f} — RSI: {c['rsi']:.0f} | Score: {c['score']:+.0f} | 7d: {c['ret7d']:+.2f}%")
        lines.append(f"")
    if sells:
        lines.append(f"### 🔴 SELL")
        for c in sells:
            lines.append(f"- **{c['ticker'].replace('-USD','')}** @ ${c['price']:,.2f} — RSI: {c['rsi']:.0f} | Score: {c['score']:+.0f} | 7d: {c['ret7d']:+.2f}%")
        lines.append(f"")
    if holds:
        lines.append(f"### ⚪ HOLD")
        for c in holds:
            lines.append(f"- **{c['ticker'].replace('-USD','')}** @ ${c['price']:,.2f} — RSI: {c['rsi']:.0f} | Score: {c['score']:+.0f}")
        lines.append(f"")

    # Weekly deep dive (Sundays only)
    if is_sun:
        lines.append(weekly_deep_dive(coins, day_num))

    lines.append(f"---")
    lines.append(f"")
    lines.append(f"*Data: Yahoo Finance + alternative.me. Educational only — not financial advice.*")
    lines.append(f"*[AICryptoPredictor](https://github.com/maic93/AICryptoPredictor) — Phase 3, Day {day_num}/90*")

    return "\n".join(lines)

# ── Logging ───────────────────────────────────────────────────────────────────

def update_signal_log(coins: list):
    log = load_json(SIGNAL_LOG, [])
    log.append({
        "date": date.today().isoformat(),
        "signals": {
            c["ticker"].replace("-USD",""): {
                "signal": c["signal"],
                "score":  c["score"],
                "price":  c["price"],
                "rsi":    c["rsi"],
                "ret1d":  c["ret1d"],
                "ret7d":  c["ret7d"],
            } for c in coins
        }
    })
    save_json(SIGNAL_LOG, log)

def update_perf_log(coins: list, day_num: int):
    log = load_json(PERF_LOG, [])
    log.append({
        "date":       date.today().isoformat(),
        "day":        day_num,
        "btc_price":  next(c["price"] for c in coins if c["ticker"]=="BTC-USD"),
        "eth_price":  next(c["price"] for c in coins if c["ticker"]=="ETH-USD"),
        "avg_ret7d":  float(np.mean([c["ret7d"]  for c in coins])),
        "avg_ret30d": float(np.mean([c["ret30d"] for c in coins])),
        "avg_rsi":    float(np.mean([c["rsi"]    for c in coins])),
        "n_buy":  sum(1 for c in coins if c["signal"]=="BUY"),
        "n_sell": sum(1 for c in coins if c["signal"]=="SELL"),
        "n_hold": sum(1 for c in coins if c["signal"]=="HOLD"),
    })
    save_json(PERF_LOG, log)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today   = date.today()
    day_num = max(1, min((today - PHASE3_START).days + 1, 90))

    print(f"Phase 3 — Enhanced Daily Market Report")
    print(f"Day {day_num}/90 | {today} | {datetime.utcnow().strftime('%H:%M UTC')}")
    print("=" * 55)

    # Fetch coin data
    coins = []
    for ticker in COINS:
        try:
            data = fetch_coin(ticker)
            coins.append(data)
            print(f"  {ticker:<12} ${data['price']:>10,.2f} | "
                  f"RSI:{data['rsi']:>5.1f} | {data['signal']} ({data['score']:+.0f})")
        except Exception as e:
            print(f"  {ticker:<12} ERROR: {e}")

    if not coins:
        print("ERROR: No data fetched!")
        return

    # Fear & Greed
    fg = fetch_fear_greed() or compute_fg_fallback(coins)
    print(f"\n  Fear & Greed : {fg['value']}/100 ({fg['label']}) [{fg['source']}]")

    # Model retraining
    print(f"  Retraining XGBoost on latest BTC data...")
    model     = retrain_and_evaluate("BTC-USD")
    drift_pct = update_model_log(model, day_num)
    if model["status"] == "ok":
        print(f"  Model MAE    : ${model['mae']:,.2f} | R2: {model['r2']:.4f}")
        if model["next_pred"]:
            pred_chg = (model["next_pred"] - model["current_price"]) / model["current_price"] * 100
            print(f"  BTC tomorrow : ${model['next_pred']:,.2f} ({pred_chg:+.2f}%)")
    if abs(drift_pct) > 10:
        print(f"  Model drift  : {drift_pct:+.1f}% vs 7d avg")

    # Build & save report
    report      = build_report(coins, fg, model, drift_pct, day_num)
    report_path = f"market_reports/{today}.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved -> {report_path}")

    # Update logs
    update_signal_log(coins)
    update_perf_log(coins, day_num)
    print(f"  Logs updated -> {SIGNAL_LOG}, {PERF_LOG}, {MODEL_PERF_LOG}")

    is_sun = today.weekday() == 6
    if is_sun:
        print(f"  Weekly deep dive included (Sunday)")

    print(f"\nDay {day_num}/90 complete!")

if __name__ == "__main__":
    main()

from datetime import date

PROJECT_START = date(2026, 6, 20)

MESSAGES = {
    1:  "📁 Day 01 — Project setup & structure",
    2:  "📥 Day 02 — Historical crypto data fetching",
    3:  "🔍 Day 03 — Exploratory Data Analysis",
    4:  "📊 Day 04 — Technical indicators (RSI, MACD, Bollinger Bands)",
    5:  "🧹 Day 05 — Data cleaning & preprocessing",
    6:  "⚙️  Day 06 — Feature engineering",
    7:  "📝 Day 07 — Week 1 recap",
    8:  "📈 Day 08 — Linear Regression baseline",
    9:  "🌳 Day 09 — Decision Tree model",
    10: "🌲 Day 10 — Random Forest + feature importance",
    11: "🚀 Day 11 — XGBoost model",
    12: "💡 Day 12 — LightGBM + cross-validation",
    13: "🏆 Day 13 — Model comparison leaderboard",
    14: "📝 Day 14 — Week 2 recap",
    15: "🧠 Day 15 — LSTM sequence prep",
    16: "🔁 Day 16 — LSTM training on BTC",
    17: "⚡ Day 17 — GRU model",
    18: "↔️  Day 18 — Bidirectional LSTM",
    19: "👁️  Day 19 — Attention mechanism",
    20: "🤖 Day 20 — Transformer model",
    21: "📝 Day 21 — Week 3 recap",
    22: "📉 Day 22 — Backtesting framework",
    23: "📐 Day 23 — Sharpe ratio & metrics",
    24: "🧩 Day 24 — Ensemble model",
    25: "⚡ Day 25 — Live prediction pipeline",
    26: "📊 Day 26 — Plotly dashboard",
    27: "✅ Day 27 — Unit tests",
    28: "🎉 Day 28 — Project complete!",
}

day = max(1, min((date.today() - PROJECT_START).days + 1, 28))
print(MESSAGES.get(day, f"🤖 Day {day} update"))

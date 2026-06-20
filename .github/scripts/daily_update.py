"""
daily_update.py — runs every day via GitHub Actions
Ticks off today's checkbox in README.md
"""
import os
import re
from datetime import date

PROJECT_START = date(2026, 6, 20)

CURRICULUM = {
    1:  "Day 01 — Project setup & structure",
    2:  "Day 02 — Fetch BTC/ETH historical data via yfinance",
    3:  "Day 03 — Exploratory Data Analysis (EDA)",
    4:  "Day 04 — Technical indicators: RSI, MACD, Bollinger Bands",
    5:  "Day 05 — Data cleaning & missing value handling",
    6:  "Day 06 — Feature engineering: lag features, rolling stats",
    7:  "Day 07 — Week 1 recap & data pipeline complete",
    8:  "Day 08 — Linear Regression baseline model",
    9:  "Day 09 — Decision Tree Regressor",
    10: "Day 10 — Random Forest + feature importance",
    11: "Day 11 — XGBoost + hyperparameter tuning",
    12: "Day 12 — LightGBM + cross-validation",
    13: "Day 13 — Model comparison leaderboard (MAE, RMSE, R²)",
    14: "Day 14 — Week 2 recap & best model saved",
    15: "Day 15 — LSTM sequence data preparation",
    16: "Day 16 — LSTM model training on BTC",
    17: "Day 17 — GRU model",
    18: "Day 18 — Bidirectional LSTM",
    19: "Day 19 — Attention mechanism",
    20: "Day 20 — Transformer-based time series model",
    21: "Day 21 — Week 3 recap & best DL model saved",
    22: "Day 22 — Backtesting framework",
    23: "Day 23 — Sharpe ratio, drawdown & strategy metrics",
    24: "Day 24 — Ensemble model (ML + DL combined)",
    25: "Day 25 — Live price fetching + real-time prediction",
    26: "Day 26 — Plotly dashboard",
    27: "Day 27 — Unit tests",
    28: "Day 28 — 🎉 Project complete!",
}

def get_day() -> int:
    return max(1, min((date.today() - PROJECT_START).days + 1, 28))

def tick_readme(day: int):
    task = CURRICULUM[day]
    readme_path = "README.md"
    with open(readme_path, "r") as f:
        content = f.read()

    # Replace the unchecked box for today's task with a checked one
    old = f"- [ ] {task}"
    new = f"- [x] {task}"
    if old in content:
        content = content.replace(old, new)
        with open(readme_path, "w") as f:
            f.write(content)
        print(f"  ✅ README updated: {task}")
    else:
        print(f"  ⚠️  Could not find checkbox for: {task}")

def run_daily_script(day: int):
    week = ((day - 1) // 7) + 1
    scripts = {
        1: "daily_progress/week1/day01setup.py",
        2: "daily_progress/week1/day02datafetch.py",
        3: "daily_progress/week1/day03eda.py",
        4: "daily_progress/week1/day04indicators.py",
        5: "daily_progress/week1/day05cleaning.py",
        6: "daily_progress/week1/day06features.py",
        7: "daily_progress/week1/day07recap.py",
        8: "daily_progress/week2/day08linearregression.py",
        9: "daily_progress/week2/day09decisiontree.py",
        10: "daily_progress/week2/day10randomforest.py",
        11: "daily_progress/week2/day11xgboost.py",
        12: "daily_progress/week2/day12lightgbm.py",
        13: "daily_progress/week2/day13comparison.py",
        14: "daily_progress/week2/day14recap.py",
        15: "daily_progress/week3/day15lstmprep.py",
        16: "daily_progress/week3/day16lstmtrain.py",
        17: "daily_progress/week3/day17gru.py",
        18: "daily_progress/week3/day18bilstm.py",
        19: "daily_progress/week3/day19attention.py",
        20: "daily_progress/week3/day20transformer.py",
        21: "daily_progress/week3/day21recap.py",
        22: "daily_progress/week4/day22backtest.py",
        23: "daily_progress/week4/day23metrics.py",
        24: "daily_progress/week4/day24ensemble.py",
        25: "daily_progress/week4/day25livepipeline.py",
        26: "daily_progress/week4/day26dashboard.py",
        27: "daily_progress/week4/day27tests.py",
        28: "daily_progress/week4/day28final.py",
    }
    script = scripts.get(day)
    if script and os.path.exists(script):
        print(f"  🚀 Running: {script}")
        os.system(f"python {script}")

def main():
    day = get_day()
    print(f"🤖 AICryptoPredictor — Day {day}/28")
    print("=" * 50)
    tick_readme(day)
    run_daily_script(day)
    print(f"\n📅 Day {day} done — see you tomorrow!")

if __name__ == "__main__":
    main()

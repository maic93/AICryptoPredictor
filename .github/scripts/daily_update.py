"""
daily_update.py — runs every day via GitHub Actions
1. Runs today's script and saves a report to reports/dayXX.md
2. Ticks today's checkbox in README.md
"""
import os
import subprocess
from datetime import date, datetime

PROJECT_START = date(2026, 6, 20)

CURRICULUM = {
    1:  ("Day 01 — Project setup & structure",                          "daily_progress/week1/day01setup.py"),
    2:  ("Day 02 — Fetch BTC/ETH historical data via yfinance",         "daily_progress/week1/day02datafetch.py"),
    3:  ("Day 03 — Exploratory Data Analysis (EDA)",                    "daily_progress/week1/day03eda.py"),
    4:  ("Day 04 — Technical indicators: RSI, MACD, Bollinger Bands",   "daily_progress/week1/day04indicators.py"),
    5:  ("Day 05 — Data cleaning & missing value handling",             "daily_progress/week1/day05cleaning.py"),
    6:  ("Day 06 — Feature engineering: lag features, rolling stats",   "daily_progress/week1/day06features.py"),
    7:  ("Day 07 — Week 1 recap & data pipeline complete",              "daily_progress/week1/day07recap.py"),
    8:  ("Day 08 — Linear Regression baseline model",                   "daily_progress/week2/day08linearregression.py"),
    9:  ("Day 09 — Decision Tree Regressor",                            "daily_progress/week2/day09decisiontree.py"),
    10: ("Day 10 — Random Forest + feature importance",                 "daily_progress/week2/day10randomforest.py"),
    11: ("Day 11 — XGBoost + hyperparameter tuning",                    "daily_progress/week2/day11xgboost.py"),
    12: ("Day 12 — LightGBM + cross-validation",                        "daily_progress/week2/day12lightgbm.py"),
    13: ("Day 13 — Model comparison leaderboard (MAE, RMSE, R²)",       "daily_progress/week2/day13comparison.py"),
    14: ("Day 14 — Week 2 recap & best model saved",                    "daily_progress/week2/day14recap.py"),
    15: ("Day 15 — LSTM sequence data preparation",                     "daily_progress/week3/day15lstmprep.py"),
    16: ("Day 16 — LSTM model training on BTC",                         "daily_progress/week3/day16lstmtrain.py"),
    17: ("Day 17 — GRU model",                                          "daily_progress/week3/day17gru.py"),
    18: ("Day 18 — Bidirectional LSTM",                                 "daily_progress/week3/day18bilstm.py"),
    19: ("Day 19 — Attention mechanism",                                "daily_progress/week3/day19attention.py"),
    20: ("Day 20 — Transformer-based time series model",                "daily_progress/week3/day20transformer.py"),
    21: ("Day 21 — Week 3 recap & best DL model saved",                 "daily_progress/week3/day21recap.py"),
    22: ("Day 22 — Backtesting framework",                              "daily_progress/week4/day22backtest.py"),
    23: ("Day 23 — Sharpe ratio, drawdown & strategy metrics",          "daily_progress/week4/day23metrics.py"),
    24: ("Day 24 — Ensemble model (ML + DL combined)",                  "daily_progress/week4/day24ensemble.py"),
    25: ("Day 25 — Live price fetching + real-time prediction",         "daily_progress/week4/day25livepipeline.py"),
    26: ("Day 26 — Plotly dashboard",                                   "daily_progress/week4/day26dashboard.py"),
    27: ("Day 27 — Unit tests",                                         "daily_progress/week4/day27tests.py"),
    28: ("Day 28 — 🎉 Project complete!",                               "daily_progress/week4/day28final.py"),
}

def get_day() -> int:
    return max(1, min((date.today() - PROJECT_START).days + 1, 28))

def run_script(script_path: str) -> str:
    """Run a daily script and capture its output."""
    if not os.path.exists(script_path):
        return f"Script not found: {script_path}"
    result = subprocess.run(
        ["python", script_path],
        capture_output=True, text=True, timeout=120
    )
    output = result.stdout
    if result.stderr:
        output += f"\nSTDERR:\n{result.stderr}"
    return output

def save_report(day: int, task: str, script: str, output: str):
    """Save a visible markdown report to reports/ so it appears in the repo."""
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/day{day:02d}.md"
    content = f"""# Day {day:02d} Report

**Task:** {task}
**Script:** `{script}`
**Date:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}

## Output

```
{output.strip()}
```
"""
    with open(report_path, "w") as f:
        f.write(content)
    print(f"  📄 Report saved → {report_path}")

def tick_readme(day: int, task: str):
    """Tick today's checkbox in README.md."""
    with open("README.md", "r") as f:
        content = f.read()
    old = f"- [ ] {task}"
    new = f"- [x] {task}"
    if old in content:
        content = content.replace(old, new)
        with open("README.md", "w") as f:
            f.write(content)
        print(f"  ✅ README ticked: {task}")
    else:
        print(f"  ⚠️  Checkbox not found for: {task}")

def main():
    day = get_day()
    task, script = CURRICULUM[day]

    print(f"🤖 AICryptoPredictor — Day {day}/28")
    print(f"📌 {task}")
    print("=" * 55)

    output = run_script(script)
    print(output)

    save_report(day, task, script, output)
    tick_readme(day, task)

    print(f"\n✅ Day {day} complete!")

if __name__ == "__main__":
    main()

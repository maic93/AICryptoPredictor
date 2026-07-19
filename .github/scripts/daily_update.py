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
    # ── WEEK 1 ────────────────────────────────────────────────────────────────
    1:  ("Day 01 — Project setup & structure",                         "daily_progress/week1/day01setup.py"),
    2:  ("Day 02 — Fetch BTC/ETH historical data via yfinance",        "daily_progress/week1/day02datafetch.py"),
    3:  ("Day 03 — Exploratory Data Analysis (EDA)",                   "daily_progress/week1/day03eda.py"),
    4:  ("Day 04 — Technical indicators: RSI, MACD, Bollinger Bands",  "daily_progress/week1/day04indicators.py"),
    5:  ("Day 05 — Data cleaning & missing value handling",            "daily_progress/week1/day05cleaning.py"),
    6:  ("Day 06 — Feature engineering: lag features, rolling stats",  "daily_progress/week1/day06features.py"),
    7:  ("Day 07 — Week 1 recap & data pipeline complete",             "daily_progress/week1/day07recap.py"),
    # ── WEEK 2 ────────────────────────────────────────────────────────────────
    8:  ("Day 08 — Linear Regression baseline model",                  "daily_progress/week2/day08linearregression.py"),
    9:  ("Day 09 — Decision Tree Regressor",                           "daily_progress/week2/day09decisiontree.py"),
    10: ("Day 10 — Random Forest + feature importance",                "daily_progress/week2/day10randomforest.py"),
    11: ("Day 11 — XGBoost + hyperparameter tuning",                   "daily_progress/week2/day11xgboost.py"),
    12: ("Day 12 — LightGBM + cross-validation",                       "daily_progress/week2/day12lightgbm.py"),
    13: ("Day 13 — Model comparison leaderboard (MAE, RMSE, R2)",      "daily_progress/week2/day13comparison.py"),
    14: ("Day 14 — Week 2 recap & best model saved",                   "daily_progress/week2/day14recap.py"),
    # ── WEEK 3 ────────────────────────────────────────────────────────────────
    15: ("Day 15 — LSTM sequence data preparation",                    "daily_progress/week3/day15lstmprep.py"),
    16: ("Day 16 — LSTM model training on BTC",                        "daily_progress/week3/day16lstmtrain.py"),
    17: ("Day 17 — GRU model",                                         "daily_progress/week3/day17gru.py"),
    18: ("Day 18 — Bidirectional LSTM",                                "daily_progress/week3/day18bilstm.py"),
    19: ("Day 19 — Attention mechanism",                               "daily_progress/week3/day19attention.py"),
    20: ("Day 20 — Transformer-based time series model",               "daily_progress/week3/day20transformer.py"),
    21: ("Day 21 — Week 3 recap & best DL model saved",                "daily_progress/week3/day21recap.py"),
    # ── WEEK 4 ────────────────────────────────────────────────────────────────
    22: ("Day 22 — Backtesting framework",                             "daily_progress/week4/day22backtest.py"),
    23: ("Day 23 — Sharpe ratio, drawdown & strategy metrics",         "daily_progress/week4/day23metrics.py"),
    24: ("Day 24 — Ensemble model (ML + DL combined)",                 "daily_progress/week4/day24ensemble.py"),
    25: ("Day 25 — Live price fetching + real-time prediction",        "daily_progress/week4/day25livepipeline.py"),
    26: ("Day 26 — Plotly dashboard",                                  "daily_progress/week4/day26dashboard.py"),
    27: ("Day 27 — Unit tests",                                        "daily_progress/week4/day27tests.py"),
    28: ("Day 28 — Week 4 complete & project summary",                 "daily_progress/week4/day28final.py"),
    # ── WEEK 5: LIVE TRADING SIGNALS ──────────────────────────────────────────
    29: ("Day 29 — Live signal engine: fetch + score all coins",       "daily_progress/week5/day29signalengine.py"),
    30: ("Day 30 — RSI + MACD combined signal strategy",               "daily_progress/week5/day30combinedsignals.py"),
    31: ("Day 31 — Signal backtesting: win rate & profit factor",      "daily_progress/week5/day31signalbacktest.py"),
    32: ("Day 32 — Confidence scoring for each signal",                "daily_progress/week5/day32confidence.py"),
    33: ("Day 33 — Multi-timeframe signal analysis (1d, 4h, 1h)",      "daily_progress/week5/day33multitimeframe.py"),
    34: ("Day 34 — Signal alert system: log & report",                 "daily_progress/week5/day34alerts.py"),
    35: ("Day 35 — Week 5 recap & signal dashboard",                   "daily_progress/week5/day35recap.py"),
    # ── WEEK 6: SENTIMENT ANALYSIS ────────────────────────────────────────────
    36: ("Day 36 — Fetch crypto news via NewsAPI / RSS",               "daily_progress/week6/day36newsfetch.py"),
    37: ("Day 37 — NLP preprocessing: tokenize, clean, stem",          "daily_progress/week6/day37nlpprep.py"),
    38: ("Day 38 — VADER sentiment scoring on crypto news",            "daily_progress/week6/day38vader.py"),
    39: ("Day 39 — Sentiment trend over time visualization",           "daily_progress/week6/day39sentimenttrend.py"),
    40: ("Day 40 — Correlate sentiment score with price movement",     "daily_progress/week6/day40correlation.py"),
    41: ("Day 41 — Sentiment-enhanced price prediction model",         "daily_progress/week6/day41sentimentmodel.py"),
    42: ("Day 42 — Week 6 recap & sentiment pipeline complete",        "daily_progress/week6/day42recap.py"),
    # ── WEEK 7: PORTFOLIO OPTIMIZER ───────────────────────────────────────────
    43: ("Day 43 — Multi-coin return & covariance matrix",             "daily_progress/week7/day43covariance.py"),
    44: ("Day 44 — Markowitz efficient frontier",                      "daily_progress/week7/day44markowitz.py"),
    45: ("Day 45 — Monte Carlo portfolio simulation",                  "daily_progress/week7/day45montecarlo.py"),
    46: ("Day 46 — Sharpe-optimal portfolio weights",                  "daily_progress/week7/day46sharpeoptimal.py"),
    47: ("Day 47 — Risk-parity portfolio allocation",                  "daily_progress/week7/day47riskparity.py"),
    48: ("Day 48 — Portfolio rebalancing strategy",                    "daily_progress/week7/day48rebalancing.py"),
    49: ("Day 49 — Week 7 recap & portfolio summary",                  "daily_progress/week7/day49recap.py"),
    # ── WEEK 8: WEB DASHBOARD ─────────────────────────────────────────────────
    50: ("Day 50 — Streamlit app structure & layout",                  "daily_progress/week8/day50streamlit.py"),
    51: ("Day 51 — Live price widget + sparklines",                    "daily_progress/week8/day51pricewidget.py"),
    52: ("Day 52 — Prediction chart component",                        "daily_progress/week8/day52predictionchart.py"),
    53: ("Day 53 — Signal dashboard component",                        "daily_progress/week8/day53signaldashboard.py"),
    54: ("Day 54 — Sentiment gauge component",                         "daily_progress/week8/day54sentimentgauge.py"),
    55: ("Day 55 — Portfolio pie chart component",                     "daily_progress/week8/day55portfoliopie.py"),
    56: ("Day 56 — Final dashboard & full project complete!",          "daily_progress/week8/day56final.py"),
}

def get_day() -> int:
    return max(1, min((date.today() - PROJECT_START).days + 1, 56))

def run_script(script_path: str) -> str:
    if not os.path.exists(script_path):
        return f"Script not found: {script_path}"
    result = subprocess.run(
        ["python", script_path],
        capture_output=True, text=True, timeout=180
    )
    output = result.stdout
    if result.stderr:
        output += f"\nSTDERR:\n{result.stderr}"
    return output

def save_report(day: int, task: str, script: str, output: str):
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
    print(f"🤖 AICryptoPredictor — Day {day}/56")
    print(f"📌 {task}")
    print("=" * 55)
    output = run_script(script)
    print(output)
    save_report(day, task, script, output)
    tick_readme(day, task)
    print(f"\n✅ Day {day} complete!")

if __name__ == "__main__":
    main()

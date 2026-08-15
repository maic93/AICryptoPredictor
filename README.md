# AICryptoPredictor

![CI](https://github.com/maic93/AICryptoPredictor/actions/workflows/ci.yml/badge.svg) ![Daily](https://github.com/maic93/AICryptoPredictor/actions/workflows/daily_update.yml/badge.svg)

> A 56-day (8-week) AI cryptocurrency system — price prediction, live trading signals, sentiment analysis, portfolio optimization & Streamlit dashboard — followed by 30 days of live daily market reports. All auto-committed via GitHub Actions.

---

## Project Phases

| Phase | Duration | Theme |
|-------|----------|-------|
| **Phase 1** | Days 1–56 (8 weeks) | Build the full AI crypto system |
| **Phase 2** | Days 57–86 (30 days) | Live daily market reports — real data, every day |

---

## Phase 1 Roadmap

| Week | Theme | Focus |
|------|-------|-------|
| Week 1 | **Foundations** | Data fetching, EDA, indicators, feature engineering |
| Week 2 | **Classical ML** | Linear Regression, Random Forest, XGBoost, LightGBM |
| Week 3 | **Deep Learning** | LSTM, GRU, Bidirectional, Attention, Transformer |
| Week 4 | **Production** | Backtesting, ensemble, live pipeline, dashboard |
| Week 5 | **Live Trading Signals** | Signal engine, multi-timeframe, backtesting, alerts |
| Week 6 | **Sentiment Analysis** | News fetch, NLP, VADER, sentiment-enhanced model |
| Week 7 | **Portfolio Optimizer** | Markowitz, Monte Carlo, Sharpe, risk-parity |
| Week 8 | **Web Dashboard** | Streamlit app, price widget, prediction, signals |

---

## Phase 1 Daily Progress

### Week 1 — Foundations
- [x] Day 01 — Project setup & structure
- [x] Day 02 — Fetch BTC/ETH historical data via yfinance
- [x] Day 03 — Exploratory Data Analysis (EDA)
- [x] Day 04 — Technical indicators: RSI, MACD, Bollinger Bands
- [x] Day 05 — Data cleaning & missing value handling
- [x] Day 06 — Feature engineering: lag features, rolling stats
- [x] Day 07 — Week 1 recap & data pipeline complete

### Week 2 — Classical ML
- [x] Day 08 — Linear Regression baseline model
- [x] Day 09 — Decision Tree Regressor
- [x] Day 10 — Random Forest + feature importance
- [x] Day 11 — XGBoost + hyperparameter tuning
- [x] Day 12 — LightGBM + cross-validation
- [x] Day 13 — Model comparison leaderboard (MAE, RMSE, R2)
- [x] Day 14 — Week 2 recap & best model saved

### Week 3 — Deep Learning
- [x] Day 15 — LSTM sequence data preparation
- [x] Day 16 — LSTM model training on BTC
- [x] Day 17 — GRU model
- [x] Day 18 — Bidirectional LSTM
- [x] Day 19 — Attention mechanism
- [x] Day 20 — Transformer-based time series model
- [x] Day 21 — Week 3 recap & best DL model saved

### Week 4 — Production
- [x] Day 22 — Backtesting framework
- [x] Day 23 — Sharpe ratio, drawdown & strategy metrics
- [x] Day 24 — Ensemble model (ML + DL combined)
- [x] Day 25 — Live price fetching + real-time prediction
- [x] Day 26 — Plotly dashboard
- [x] Day 27 — Unit tests
- [x] Day 28 — Week 4 complete & project summary

### Week 5 — Live Trading Signals
- [x] Day 29 — Live signal engine: fetch + score all coins
- [x] Day 30 — RSI + MACD combined signal strategy
- [x] Day 31 — Signal backtesting: win rate & profit factor
- [x] Day 32 — Confidence scoring for each signal
- [x] Day 33 — Multi-timeframe signal analysis
- [x] Day 34 — Signal alert system: log & report
- [x] Day 35 — Week 5 recap & signal dashboard

### Week 6 — Sentiment Analysis
- [x] Day 36 — Fetch crypto news via RSS
- [x] Day 37 — NLP preprocessing: tokenize, clean, stem
- [x] Day 38 — VADER sentiment scoring on crypto news
- [x] Day 39 — Sentiment trend over time visualization
- [x] Day 40 — Correlate sentiment score with price movement
- [x] Day 41 — Sentiment-enhanced price prediction model
- [x] Day 42 — Week 6 recap & sentiment pipeline complete

### Week 7 — Portfolio Optimizer
- [x] Day 43 — Multi-coin return & covariance matrix
- [x] Day 44 — Markowitz efficient frontier
- [x] Day 45 — Monte Carlo portfolio simulation
- [x] Day 46 — Sharpe-optimal portfolio weights
- [x] Day 47 — Risk-parity portfolio allocation
- [x] Day 48 — Portfolio rebalancing strategy
- [x] Day 49 — Week 7 recap & portfolio summary

### Week 8 — Web Dashboard
- [x] Day 50 — Streamlit app structure & layout
- [x] Day 51 — Live price widget + sparklines
- [x] Day 52 — Prediction chart component
- [x] Day 53 — Signal dashboard component
- [x] Day 54 — Sentiment gauge component
- [x] Day 55 — Portfolio pie chart component
- [x] Day 56 — Final dashboard & full project complete!

---

## Phase 2 — Live Daily Market Reports

Every day a new markdown report is auto-committed to `market_reports/` with:

- **Live prices** for BTC, ETH, SOL, BNB, ADA, DOT
- **RSI, MACD, Bollinger Bands** — full technical analysis table
- **Buy/Sell/Hold signals** with confidence scores
- **Fear & Greed Index** — daily market mood
- **OHLC table** — open, high, low, close for each coin
- **Market narrative** — plain English summary of conditions
- **Running logs** in `data/signal_log.json` and `data/performance_log.json`

No images — pure markdown, zero repo bloat, real historical data.

### Sample Report Structure
```
market_reports/
├── 2026-08-14.md   ← Day 1
├── 2026-08-15.md   ← Day 2
├── ...
└── 2026-09-12.md   ← Day 30
```

---

## Repository Structure

```
AICryptoPredictor/
├── .github/
│   ├── workflows/           # daily_update.yml, ci.yml, rerun_day.yml
│   └── scripts/             # daily_update.py, get_commit_message.py, rerun_day.py
├── daily_progress/
│   ├── week1/ — week8/      # 56 daily Python scripts (Phase 1)
│   └── phase2/              # daily_market_report.py (Phase 2)
├── market_reports/          # Daily .md reports (Phase 2, auto-generated)
├── reports/                 # Phase 1 daily reports + charts
├── data/                    # signal_log.json, performance_log.json, CSV datasets
├── dashboard/               # Streamlit app (Week 8)
├── models/                  # Saved trained models
├── tests/                   # CI test suite (18 tests)
└── main.py
```

---

## Quick Start

```bash
git clone https://github.com/maic93/AICryptoPredictor.git
cd AICryptoPredictor
pip install -r requirements.txt
python main.py
```

## Tech Stack

`pandas` · `numpy` · `yfinance` · `scikit-learn` · `xgboost` · `lightgbm` · `matplotlib` · `plotly` · `streamlit` · `scipy` · `requests`

## Coins Tracked

BTC · ETH · SOL · BNB · ADA · DOT

---

> **Disclaimer**: Educational purposes only. Not financial advice.

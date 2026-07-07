# 🤖📈 AICryptoPredictor

> A 28-day progressive project building an AI-powered cryptocurrency price prediction system using Python and Machine Learning — auto-committed daily via GitHub Actions.

---

## 🗓️ Roadmap

| Week | Theme | Focus |
|------|-------|-------|
| Week 1 | **Foundations** | Data fetching, EDA, indicators, feature engineering |
| Week 2 | **Classical ML** | Linear Regression, Random Forest, XGBoost, LightGBM |
| Week 3 | **Deep Learning** | LSTM, GRU, Bidirectional, Attention, Transformer |
| Week 4 | **Production** | Backtesting, ensemble, live pipeline, dashboard |

---

## 📊 Daily Progress

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
- [x] Day 13 — Model comparison leaderboard (MAE, RMSE, R²)
- [x] Day 14 — Week 2 recap & best model saved

### Week 3 — Deep Learning
- [x] Day 15 — LSTM sequence data preparation
- [x] Day 16 — LSTM model training on BTC
- [x] Day 17 — GRU model
- [x] Day 18 — Bidirectional LSTM
- [ ] Day 19 — Attention mechanism
- [ ] Day 20 — Transformer-based time series model
- [ ] Day 21 — Week 3 recap & best DL model saved

### Week 4 — Production
- [ ] Day 22 — Backtesting framework
- [ ] Day 23 — Sharpe ratio, drawdown & strategy metrics
- [ ] Day 24 — Ensemble model (ML + DL combined)
- [ ] Day 25 — Live price fetching + real-time prediction
- [ ] Day 26 — Plotly dashboard
- [ ] Day 27 — Unit tests
- [x] Day 28 — 🎉 Project complete!

---

## 📁 Structure

```
AICryptoPredictor/
├── .github/
│   ├── workflows/        # GitHub Actions daily automation
│   └── scripts/          # Daily update & commit message scripts
├── daily_progress/       # Auto-generated daily Python files
│   ├── week1/
│   ├── week2/
│   ├── week3/
│   └── week4/
├── models/               # Saved trained models
├── data/                 # Raw and processed datasets
├── utils/                # Shared helper functions
├── tests/                # Unit tests
└── main.py               # Entry point
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/AICryptoPredictor.git
cd AICryptoPredictor
pip install -r requirements.txt
python main.py
```

## 📦 Tech Stack

`pandas` · `numpy` · `yfinance` · `scikit-learn` · `xgboost` · `lightgbm` · `tensorflow` · `keras` · `matplotlib` · `plotly` · `backtrader` · `ta`

## 💰 Coins Tracked

| Coin | Symbol |
|------|--------|
| Bitcoin | BTC-USD |
| Ethereum | ETH-USD |
| Solana | SOL-USD |
| Binance Coin | BNB-USD |

---

> ⚠️ **Disclaimer**: Educational purposes only. Not financial advice.

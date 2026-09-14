from datetime import date

PROJECT_START = date(2026, 6, 20)
PHASE2_START  = date(2026, 8, 16)
PHASE3_START  = date(2026, 9, 15)  # UPDATE to your actual Phase 3 start date

PHASE1_MESSAGES = {
    1:"Day 01 - Project setup & structure",
    2:"Day 02 - Historical crypto data fetching",
    3:"Day 03 - Exploratory Data Analysis",
    4:"Day 04 - Technical indicators (RSI, MACD, BB)",
    5:"Day 05 - Data cleaning & preprocessing",
    6:"Day 06 - Feature engineering",
    7:"Day 07 - Week 1 recap",
    8:"Day 08 - Linear Regression baseline",
    9:"Day 09 - Decision Tree model",
    10:"Day 10 - Random Forest + feature importance",
    11:"Day 11 - XGBoost model",
    12:"Day 12 - LightGBM + cross-validation",
    13:"Day 13 - Model comparison leaderboard",
    14:"Day 14 - Week 2 recap",
    15:"Day 15 - LSTM sequence prep",
    16:"Day 16 - LSTM training on BTC",
    17:"Day 17 - GRU model",
    18:"Day 18 - Bidirectional LSTM",
    19:"Day 19 - Attention mechanism",
    20:"Day 20 - Transformer model",
    21:"Day 21 - Week 3 recap",
    22:"Day 22 - Backtesting framework",
    23:"Day 23 - Sharpe ratio & metrics",
    24:"Day 24 - Ensemble model",
    25:"Day 25 - Live prediction pipeline",
    26:"Day 26 - Plotly dashboard",
    27:"Day 27 - Unit tests",
    28:"Day 28 - Phase 1 week 4 complete",
    29:"Day 29 - Live signal engine",
    30:"Day 30 - Combined RSI + MACD signals",
    31:"Day 31 - Signal backtesting",
    32:"Day 32 - Signal confidence scoring",
    33:"Day 33 - Multi-timeframe analysis",
    34:"Day 34 - Signal alert system",
    35:"Day 35 - Week 5 recap",
    36:"Day 36 - Crypto news fetching",
    37:"Day 37 - NLP preprocessing",
    38:"Day 38 - VADER sentiment scoring",
    39:"Day 39 - Sentiment trend visualization",
    40:"Day 40 - Sentiment vs price correlation",
    41:"Day 41 - Sentiment-enhanced prediction",
    42:"Day 42 - Week 6 recap",
    43:"Day 43 - Multi-coin covariance matrix",
    44:"Day 44 - Markowitz efficient frontier",
    45:"Day 45 - Monte Carlo simulation",
    46:"Day 46 - Sharpe-optimal portfolio",
    47:"Day 47 - Risk-parity allocation",
    48:"Day 48 - Portfolio rebalancing",
    49:"Day 49 - Week 7 recap",
    50:"Day 50 - Streamlit app structure",
    51:"Day 51 - Live price widget",
    52:"Day 52 - Prediction chart component",
    53:"Day 53 - Signal dashboard component",
    54:"Day 54 - Sentiment gauge component",
    55:"Day 55 - Portfolio pie chart",
    56:"Day 56 - Full 8-week project complete",
}

today = date.today()

if today >= PHASE3_START:
    day_num = max(1, min((today - PHASE3_START).days + 1, 90))
    is_sun  = today.weekday() == 6
    week    = (day_num - 1) // 7 + 1
    if is_sun:
        print(f"Phase 3 Day {day_num}/90 - Weekly deep dive + live signals (Week {week})")
    else:
        print(f"Phase 3 Day {day_num}/90 - Live signals, Fear & Greed, model retrain")

elif today >= PHASE2_START:
    day_num = max(1, min((today - PHASE2_START).days + 1, 30))
    print(f"Phase 2 Day {day_num}/30 - Live market report")

else:
    day = max(1, min((today - PROJECT_START).days + 1, 56))
    print(PHASE1_MESSAGES.get(day, f"Day {day} update"))

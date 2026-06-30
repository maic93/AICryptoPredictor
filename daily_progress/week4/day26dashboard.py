"""
Day 26 — Plotly Dashboard: Price Chart + Prediction Overlay
Week 4: Production
"""
import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

def get_predictions(ticker="BTC-USD"):
    df = yf.download(ticker, period="1y", progress=False, auto_adjust=True).dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    close = df["Close"].squeeze()
    for lag in [1, 2, 3, 5, 7]:
        df[f"lag{lag}"] = close.shift(lag)
    for w in [7, 14]:
        df[f"rollMean{w}"] = close.rolling(w).mean()
    df["return1d"] = close.pct_change()
    df["target"] = close.shift(-1)
    df.dropna(inplace=True)
    feat_cols = [c for c in df.columns if c not in ["target","Open","High","Low","Close","Volume"]]
    X, y = df[feat_cols].values, df["target"].values
    split = int(len(X) * 0.8)
    model = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    model.fit(X[:split], y[:split])
    preds = model.predict(X[split:])
    return df.index[split:], close.values[split:], preds

def run():
    print("📊 Day 26 — Plotly Dashboard")
    print("=" * 50)
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        dates, actual, predicted = get_predictions("BTC-USD")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("BTC Price: Actual vs Predicted", "Prediction Error"),
                            row_heights=[0.7, 0.3])

        fig.add_trace(go.Scatter(x=dates, y=actual, name="Actual",
                                  line=dict(color="#F7931A", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=predicted, name="Predicted",
                                  line=dict(color="#627EEA", width=2, dash="dash")), row=1, col=1)
        error = predicted - actual
        fig.add_trace(go.Bar(x=dates, y=error, name="Error",
                              marker_color=["#e74c3c" if e < 0 else "#2ecc71" for e in error]), row=2, col=1)

        fig.update_layout(title="🤖 AICryptoPredictor — BTC Dashboard",
                          template="plotly_dark", height=700,
                          legend=dict(orientation="h"))
        fig.write_html("data/dashboard.html")
        print("  ✅ Dashboard saved → data/dashboard.html")
        print("  Open it in your browser to view!")

    except ImportError:
        print("  ⚠️  Plotly not installed: pip install plotly")

    print("\n✅ Day 26 complete!")

if __name__ == "__main__":
    run()

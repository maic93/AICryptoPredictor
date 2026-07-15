"""
Day 26 — Plotly Dashboard: Price Chart + Prediction Overlay
Week 4: Production
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import os

os.makedirs("reports", exist_ok=True)

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
    feat_cols = [str(c) for c in df.columns
                 if c not in ["target","Open","High","Low","Close","Volume"]]
    df.columns = [str(c) for c in df.columns]
    X = df[feat_cols]
    y = df["target"]
    close_clean = df["Close"].squeeze()
    split = int(len(X) * 0.8)
    model = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    model.fit(X.iloc[:split], y.iloc[:split])
    preds  = model.predict(X.iloc[split:])
    # Use iloc[split:] from close_clean to guarantee same length
    actual = close_clean.iloc[split:].values
    dates  = df.index[split:]
    # Safety trim to shortest
    n = min(len(preds), len(actual), len(dates))
    return dates[:n], actual[:n], preds[:n]

def plot_matplotlib(dates, actual, predicted):
    """Fallback matplotlib chart (always works on GitHub Actions)"""
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    axes[0].plot(dates, actual,    color="#F7931A", linewidth=1.5, label="Actual")
    axes[0].plot(dates, predicted, color="#627EEA", linewidth=1.5,
                 linestyle="--", label="Predicted")
    axes[0].set_title("BTC Price: Actual vs Predicted", fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    error = predicted - actual
    axes[1].bar(dates, error,
                color=["#e74c3c" if e < 0 else "#2ecc71" for e in error],
                width=1, alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Prediction Error (USD)", fontweight="bold")
    axes[1].set_ylabel("Error"); axes[1].grid(alpha=0.3)
    plt.suptitle("Day 26 - AICryptoPredictor Dashboard", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day26chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("Day 26 - Plotly Dashboard")
    print("=" * 50)
    dates, actual, predicted = get_predictions("BTC-USD")
    print(f"  Test samples : {len(actual)}")
    print(f"  Date range   : {dates[0].date()} → {dates[-1].date()}")

    mae = np.mean(np.abs(predicted - actual))
    print(f"  MAE          : ${mae:,.2f}")

    # Try Plotly first, fallback to matplotlib
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("BTC Price: Actual vs Predicted",
                                            "Prediction Error"),
                            row_heights=[0.7, 0.3])
        fig.add_trace(go.Scatter(x=dates, y=actual, name="Actual",
                                  line=dict(color="#F7931A", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=predicted, name="Predicted",
                                  line=dict(color="#627EEA", width=2, dash="dash")),
                      row=1, col=1)
        error = predicted - actual
        fig.add_trace(go.Bar(x=dates, y=error, name="Error",
                              marker_color=["#e74c3c" if e < 0 else "#2ecc71"
                                            for e in error]), row=2, col=1)
        fig.update_layout(title="AICryptoPredictor - BTC Dashboard",
                          template="plotly_dark", height=700)
        fig.write_html("reports/day26dashboard.html")
        print("  ✅ Plotly dashboard saved → reports/day26dashboard.html")
    except Exception as e:
        print(f"  Plotly skipped ({e}), using matplotlib instead.")

    plot_matplotlib(dates, actual, predicted)
    print("\n✅ Day 26 complete!")

if __name__ == "__main__":
    run()

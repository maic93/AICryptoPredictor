"""
Day 08 — Linear Regression Baseline
Week 2: Classical ML
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

os.makedirs("reports", exist_ok=True)

def build_dataset(ticker="BTC-USD"):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True).dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    close = df["Close"].squeeze()
    for lag in [1, 2, 3, 5, 7, 14]:
        df[f"lag{lag}"] = close.shift(lag)
    for w in [7, 14, 30]:
        df[f"rollMean{w}"] = close.rolling(w).mean()
        df[f"rollStd{w}"]  = close.rolling(w).std()
    df["return1d"] = close.pct_change()
    df["return7d"] = close.pct_change(7)
    df["target"] = close.shift(-1)
    df.dropna(inplace=True)
    feat_cols = [c for c in df.columns if c not in ["target","Open","High","Low","Close","Volume"]]
    return df[feat_cols].values, df["target"].values, df.index

def plot(y_test, preds, dates):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    axes[0].plot(dates, y_test, color="#F7931A", linewidth=1.5, label="Actual")
    axes[0].plot(dates, preds,  color="#3498DB", linewidth=1.5, linestyle="--", label="Predicted")
    axes[0].set_title("Linear Regression — Actual vs Predicted", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    error = preds - y_test
    axes[1].bar(dates, error, color=["#E74C3C" if e < 0 else "#2ECC71" for e in error], width=1, alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Prediction Error", fontweight="bold")
    axes[1].set_ylabel("Error (USD)")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = "reports/day08chart.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("📈 Day 08 — Linear Regression Baseline")
    print("=" * 50)
    X, y, idx = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    dates_test = idx[-len(y_test):]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  R²   : {r2:.4f}")
    plot(y_test, preds, dates_test)
    print("\n✅ Linear Regression complete!")

if __name__ == "__main__":
    run()

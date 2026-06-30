"""
Day 10 — Random Forest + Feature Importance
Week 2: Classical ML
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

os.makedirs("reports", exist_ok=True)

def build_dataset(ticker="BTC-USD"):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True).dropna()
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    close = df["Close"].squeeze()
    for lag in [1, 2, 3, 5, 7, 14]:
        df[f"lag{lag}"] = close.shift(lag)
    for w in [7, 14, 30]:
        df[f"rollMean{w}"] = close.rolling(w).mean()
        df[f"rollStd{w}"]  = close.rolling(w).std()
    df["return1d"] = close.pct_change()
    df["return7d"]  = close.pct_change(7)
    df["target"] = close.shift(-1)
    df.dropna(inplace=True)
    feat_cols = [c for c in df.columns if c not in ["target","Open","High","Low","Close","Volume"]]
    # Ensure all column names are plain strings
    feat_cols = [str(c) for c in feat_cols]
    df.columns = [str(c) for c in df.columns]
    return df[feat_cols].values, df["target"].values, feat_cols, df.index

def plot(y_test, preds, dates, feat_cols, importances):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    axes[0].plot(dates, y_test, color="#F7931A", linewidth=1.5, label="Actual")
    axes[0].plot(dates, preds,  color="#2ECC71", linewidth=1.5, linestyle="--", label="Predicted")
    axes[0].set_title("Random Forest — Actual vs Predicted", fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    sorted_idx = np.argsort(importances)[-15:]
    axes[1].barh([feat_cols[i] for i in sorted_idx],
                  [importances[i] for i in sorted_idx], color="#3498DB")
    axes[1].set_title("Top 15 Feature Importances", fontweight="bold")
    axes[1].set_xlabel("Importance Score")
    axes[1].grid(alpha=0.3, axis="x")
    plt.tight_layout()
    out = "reports/day10chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("🌲 Day 10 — Random Forest + Feature Importance")
    print("=" * 50)
    X, y, feat_cols, idx = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    dates_test = idx[-len(y_test):]
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  R²   : {r2:.4f}")
    print("\n  Top 5 Features:")
    importances = model.feature_importances_
    for name, score in sorted(zip(feat_cols, importances), key=lambda x: -x[1])[:5]:
        print(f"    {str(name):>15s}: {'█' * int(score*100)} ({score:.4f})")
    plot(y_test, preds, dates_test, feat_cols, importances)
    print("\n✅ Random Forest complete!")

if __name__ == "__main__":
    run()

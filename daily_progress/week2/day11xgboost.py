"""
Day 11 — XGBoost + Hyperparameter Tuning
Week 2: Classical ML
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

os.makedirs("reports", exist_ok=True)

def build_dataset(ticker="BTC-USD"):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True).dropna()
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

def plot(y_test, preds, dates, cv_results):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    axes[0].plot(dates, y_test, color="#F7931A", linewidth=1.5, label="Actual")
    axes[0].plot(dates, preds,  color="#E74C3C", linewidth=1.5, linestyle="--", label="XGBoost Predicted")
    axes[0].set_title("XGBoost — Actual vs Predicted", fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    params = [f"n={p['n_estimators']}\nlr={p['learning_rate']}\nd={p['max_depth']}"
              for p in cv_results["params"]]
    scores = -cv_results["mean_test_score"]
    colors = ["#2ECC71" if s == min(scores) else "#3498DB" for s in scores]
    axes[1].bar(range(len(scores)), scores, color=colors)
    axes[1].set_xticks(range(len(params)))
    axes[1].set_xticklabels(params, fontsize=7)
    axes[1].set_title("GridSearch CV — MAE per Parameter Combo (lower = better)", fontweight="bold")
    axes[1].set_ylabel("MAE (USD)"); axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = "reports/day11chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("🚀 Day 11 — XGBoost + Hyperparameter Tuning")
    print("=" * 50)
    X, y, idx = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    dates_test = idx[-len(y_test):]
    param_grid = {"n_estimators": [100, 300], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]}
    base = XGBRegressor(random_state=42, verbosity=0)
    grid = GridSearchCV(base, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    preds = grid.best_estimator_.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    print(f"  Best params : {grid.best_params_}")
    print(f"  MAE         : ${mae:,.2f}")
    print(f"  RMSE        : ${rmse:,.2f}")
    print(f"  R²          : {r2:.4f}")
    plot(y_test, preds, dates_test, grid.cv_results_)
    print("\n✅ XGBoost complete!")

if __name__ == "__main__":
    run()

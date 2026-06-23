"""
Day 14 — Week 2 Recap & Save Best Model
Week 2: Classical ML
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle, os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("models", exist_ok=True)
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

def plot(y_test, preds, dates):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    axes[0].plot(dates, y_test, color="#F7931A", linewidth=1.5, label="Actual")
    axes[0].plot(dates, preds,  color="#E74C3C", linewidth=1.5, linestyle="--", label="XGBoost (Best)")
    axes[0].set_title("Week 2 Best Model — XGBoost Predictions", fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    error_pct = (preds - y_test) / y_test * 100
    axes[1].fill_between(dates, error_pct, alpha=0.4,
                          color="#E74C3C" if error_pct.mean() < 0 else "#2ECC71")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Percentage Error (%)", fontweight="bold")
    axes[1].set_ylabel("Error %"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out = "reports/day14chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("📝 Day 14 — Week 2 Recap")
    print("=" * 50)
    X, y, idx = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    dates_test = idx[-len(y_test):]
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"  Final MAE : ${mean_absolute_error(y_test, preds):,.2f}")
    print(f"  Final R²  : {r2_score(y_test, preds):.4f}")
    with open("models/best_classical.pkl", "wb") as f:
        pickle.dump(model, f)
    print("  💾 Model saved → models/best_classical.pkl")
    plot(y_test, preds, dates_test)
    print("\n🎉 Week 2 done! Next: Deep Learning")

if __name__ == "__main__":
    run()

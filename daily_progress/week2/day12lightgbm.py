"""
Day 12 — LightGBM + Cross-Validation
Week 2: Classical ML
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
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
    return df[feat_cols].values, df["target"].values

def plot(maes, rmses, r2s):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    folds = [f"Fold {i+1}" for i in range(len(maes))]
    for ax, vals, label, color in zip(
        axes,
        [maes, rmses, r2s],
        ["MAE (USD)", "RMSE (USD)", "R²"],
        ["#3498DB", "#E74C3C", "#2ECC71"]
    ):
        bars = ax.bar(folds, vals, color=color, alpha=0.8)
        ax.axhline(np.mean(vals), color="black", linestyle="--", linewidth=1,
                   label=f"Mean: {np.mean(vals):,.2f}")
        ax.set_title(label, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{v:,.0f}" if label != "R²" else f"{v:.3f}",
                    ha="center", fontsize=8)
    plt.suptitle("LightGBM — 5-Fold Time Series Cross-Validation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day12chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("💡 Day 12 — LightGBM + Cross-Validation")
    print("=" * 50)
    X, y = build_dataset()
    tscv = TimeSeriesSplit(n_splits=5)
    maes, rmses, r2s = [], [], []
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                   num_leaves=31, random_state=42, verbose=-1)
        model.fit(X[tr], y[tr])
        preds = model.predict(X[te])
        maes.append(mean_absolute_error(y[te], preds))
        rmses.append(np.sqrt(mean_squared_error(y[te], preds)))
        r2s.append(r2_score(y[te], preds))
        print(f"  Fold {fold}: MAE=${maes[-1]:>9,.2f} | RMSE=${rmses[-1]:>9,.2f} | R²={r2s[-1]:.4f}")
    print(f"\n  Mean MAE  : ${np.mean(maes):,.2f} ± ${np.std(maes):,.2f}")
    print(f"  Mean RMSE : ${np.mean(rmses):,.2f} ± ${np.std(rmses):,.2f}")
    print(f"  Mean R²   : {np.mean(r2s):.4f}")
    plot(maes, rmses, r2s)
    print("\n✅ LightGBM complete!")

if __name__ == "__main__":
    run()

"""
Day 09 — Decision Tree Regressor
Week 2: Classical ML
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

def plot(y_test, results, dates):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    best_depth, best_preds = min(results, key=lambda x: x[1]), None
    for depth, mae, preds in results:
        if mae == min(r[1] for r in results):
            best_preds = preds
    axes[0].plot(dates, y_test,      color="#F7931A", linewidth=1.5, label="Actual")
    axes[0].plot(dates, best_preds,  color="#9B59B6", linewidth=1.5, linestyle="--", label=f"Best Tree (depth={results[0][0]})")
    axes[0].set_title("Decision Tree — Best Model vs Actual", fontweight="bold")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    depths = [str(r[0]) for r in results]
    maes   = [r[1] for r in results]
    bars = axes[1].bar(depths, maes, color=["#2ECC71" if m == min(maes) else "#3498DB" for m in maes])
    axes[1].set_title("MAE by Tree Depth (lower = better)", fontweight="bold")
    axes[1].set_xlabel("Max Depth"); axes[1].set_ylabel("MAE (USD)")
    for bar, mae in zip(bars, maes):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     f"${mae:,.0f}", ha="center", fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = "reports/day09chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("🌳 Day 09 — Decision Tree Regressor")
    print("=" * 50)
    X, y, idx = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    dates_test = idx[-len(y_test):]
    results = []
    for depth in [3, 5, 10, None]:
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        label = str(depth) if depth else "None"
        results.append((label, mae, preds))
        print(f"  depth={label:>4s} | MAE=${mae:>9,.2f} | RMSE=${rmse:>9,.2f} | R²={r2:.4f}")
    plot(y_test, results, dates_test)
    print("\n✅ Decision Tree complete!")

if __name__ == "__main__":
    run()

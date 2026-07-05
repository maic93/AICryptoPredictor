"""
Day 13 — Model Comparison Leaderboard
Week 2: Classical ML
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb
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
    df["return7d"]  = close.pct_change(7)
    df["target"] = close.shift(-1)
    df.dropna(inplace=True)
    feat_cols = [str(c) for c in df.columns
                 if c not in ["target","Open","High","Low","Close","Volume"]]
    df.columns = [str(c) for c in df.columns]
    return df[feat_cols], df["target"], df.index

def plot(results, y_test, all_preds, dates):
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    names  = [r["Model"] for r in results]
    maes   = [r["MAE"]   for r in results]
    colors = ["#F1C40F" if i == 0 else "#3498DB" for i in range(len(names))]
    bars = axes[0].barh(names[::-1], maes[::-1], color=colors[::-1])
    axes[0].set_title("Model Leaderboard — MAE (lower = better) [BEST]", fontweight="bold")
    axes[0].set_xlabel("MAE (USD)")
    for bar, mae in zip(bars, maes[::-1]):
        axes[0].text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                     f"${mae:,.0f}", va="center", fontsize=9)
    axes[0].grid(alpha=0.3, axis="x")

    axes[1].plot(dates, y_test, color="#F7931A", linewidth=2, label="Actual", zorder=5)
    line_colors = ["#3498DB","#2ECC71","#9B59B6","#E74C3C","#1ABC9C"]
    for (name, preds), color in zip(all_preds.items(), line_colors):
        axes[1].plot(dates, preds, linewidth=1, linestyle="--",
                     color=color, label=name, alpha=0.8)
    axes[1].set_title("All Models — Predictions vs Actual", fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = "reports/day13chart.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("Day 13 — Model Comparison Leaderboard")
    print("=" * 60)
    X, y, idx = build_dataset()
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    dates_test = idx[-len(y_test):]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "LinearRegression": (LinearRegression(), True),
        "DecisionTree":     (DecisionTreeRegressor(max_depth=5, random_state=42), False),
        "RandomForest":     (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), False),
        "XGBoost":          (XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=0), False),
        "LightGBM":         (lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1), False),
    }

    results, all_preds = [], {}
    for name, (model, scaled) in models.items():
        if scaled:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
        else:
            # Pass DataFrame so LightGBM/RF have feature names
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        all_preds[name] = preds
        results.append({
            "Model": name,
            "MAE":  mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2":   r2_score(y_test, preds),
        })

    results.sort(key=lambda x: x["MAE"])
    print(f"\n  {'Rank':<5} {'Model':<20} {'MAE':>12} {'RMSE':>12} {'R2':>8}")
    print("  " + "-" * 60)
    for i, r in enumerate(results, 1):
        crown = " [BEST]" if i == 1 else ""
        print(f"  {i:<5} {r['Model']:<20} ${r['MAE']:>10,.2f} ${r['RMSE']:>10,.2f} {r['R2']:>8.4f}{crown}")

    plot(results, y_test.values, all_preds, dates_test)
    print(f"\n  Best model: {results[0]['Model']}")
    print("\n✅ Day 13 complete!")

if __name__ == "__main__":
    run()

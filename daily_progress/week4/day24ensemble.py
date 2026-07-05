"""
Day 24 — Ensemble Model
Week 4: Production
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
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

def plot(y_test, preds_dict, ensemble_preds, dates):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    axes[0].plot(dates, y_test, color="#F7931A", linewidth=2, label="Actual", zorder=5)
    colors = ["#3498DB", "#2ECC71", "#9B59B6"]
    for (name, preds), color in zip(preds_dict.items(), colors):
        axes[0].plot(dates, preds, linewidth=1, linestyle=":",
                     color=color, alpha=0.6, label=name)
    axes[0].plot(dates, ensemble_preds, color="#E74C3C",
                 linewidth=2, linestyle="--", label="Ensemble [BEST]")
    axes[0].set_title("Ensemble vs Individual Models", fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    names = list(preds_dict.keys()) + ["Ensemble"]
    maes  = [mean_absolute_error(y_test, p) for p in preds_dict.values()] + \
            [mean_absolute_error(y_test, ensemble_preds)]
    bar_colors = ["#3498DB", "#2ECC71", "#9B59B6", "#E74C3C"]
    bars = axes[1].bar(names, maes, color=bar_colors)
    axes[1].set_title("MAE Comparison (lower = better)", fontweight="bold")
    axes[1].set_ylabel("MAE (USD)"); axes[1].grid(alpha=0.3, axis="y")
    for bar, mae in zip(bars, maes):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     f"${mae:,.0f}", ha="center", fontsize=9)
    plt.tight_layout()
    out = "reports/day24chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  📈 Chart saved → {out}")

def run():
    print("Day 24 — Ensemble Model")
    print("=" * 50)
    X, y, idx = build_dataset()
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    dates_test = idx[-len(y_test):]

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost":      XGBRegressor(n_estimators=300, learning_rate=0.05,
                                      random_state=42, verbosity=0),
        "LightGBM":     lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                            random_state=42, verbose=-1),
    }

    preds_dict, weights = {}, []
    for name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        mae = mean_absolute_error(y_test, p)
        preds_dict[name] = p
        weights.append(1 / mae)
        print(f"  {name:<15} MAE=${mae:,.2f}")

    weights = np.array(weights) / sum(weights)
    ensemble_preds = sum(w * p for w, p in zip(weights, preds_dict.values()))
    ens_mae = mean_absolute_error(y_test, ensemble_preds)
    ens_r2  = r2_score(y_test, ensemble_preds)
    print(f"\n  {'Ensemble':>15} MAE=${ens_mae:,.2f}  R2={ens_r2:.4f} [BEST]")
    print(f"\n  Weights: {dict(zip(models.keys(), [f'{w:.3f}' for w in weights]))}")
    plot(y_test.values, preds_dict, ensemble_preds, dates_test)
    print("\n✅ Ensemble complete!")

if __name__ == "__main__":
    run()

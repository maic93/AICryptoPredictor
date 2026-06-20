"""
Day 24 — Ensemble Model: Combine ML + DL Predictions
Week 4: Production
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

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
    feat_cols = [c for c in df.columns if c not in ["target", "Open", "High", "Low", "Close", "Volume"]]
    return df[feat_cols].values, df["target"].values

def run():
    print("🧩 Day 24 — Ensemble Model")
    print("=" * 50)
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    models = {
        "RandomForest" : RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost"      : XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=0),
        "LightGBM"     : lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1),
    }

    preds_list, weights = [], []
    for name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        mae = mean_absolute_error(y_test, p)
        preds_list.append(p)
        weights.append(1 / mae)  # weight inversely proportional to error
        print(f"  {name:<15} MAE=${mae:,.2f}")

    # Weighted average ensemble
    weights = np.array(weights) / sum(weights)
    ensemble_preds = sum(w * p for w, p in zip(weights, preds_list))
    ens_mae = mean_absolute_error(y_test, ensemble_preds)
    ens_r2  = r2_score(y_test, ensemble_preds)

    print(f"\n  {'Ensemble':>15} MAE=${ens_mae:,.2f}  R²={ens_r2:.4f}  👑")
    print(f"\n  Weights used: {dict(zip(models.keys(), [f'{w:.3f}' for w in weights]))}")
    print("\n✅ Ensemble model complete!")

if __name__ == "__main__":
    run()

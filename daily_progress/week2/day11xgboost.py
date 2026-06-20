"""
Day 11 — XGBoost + Hyperparameter Tuning
Week 2: Classical ML
"""
import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    print("🚀 Day 11 — XGBoost + Hyperparameter Tuning")
    print("=" * 50)
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    }
    base = XGBRegressor(random_state=42, verbosity=0)
    grid = GridSearchCV(base, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    preds = best.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    print(f"  Best params : {grid.best_params_}")
    print(f"  MAE         : ${mae:,.2f}")
    print(f"  RMSE        : ${rmse:,.2f}")
    print(f"  R²          : {r2:.4f}")
    print("\n✅ XGBoost tuning complete!")

if __name__ == "__main__":
    run()

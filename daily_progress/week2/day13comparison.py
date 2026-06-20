"""
Day 13 — Model Comparison Leaderboard
Week 2: Classical ML
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    print("🏆 Day 13 — Model Comparison Leaderboard")
    print("=" * 60)
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "LinearRegression" : (LinearRegression(), True),
        "DecisionTree"     : (DecisionTreeRegressor(max_depth=5, random_state=42), False),
        "RandomForest"     : (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), False),
        "XGBoost"          : (XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=0), False),
        "LightGBM"         : (lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1), False),
    }

    results = []
    for name, (model, scaled) in models.items():
        Xtr = X_train_s if scaled else X_train
        Xte = X_test_s  if scaled else X_test
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds),
        })

    results.sort(key=lambda x: x["MAE"])
    print(f"\n  {'Rank':<5} {'Model':<20} {'MAE':>12} {'RMSE':>12} {'R²':>8}")
    print("  " + "-" * 60)
    for i, r in enumerate(results, 1):
        crown = " 👑" if i == 1 else ""
        print(f"  {i:<5} {r['Model']:<20} ${r['MAE']:>10,.2f} ${r['RMSE']:>10,.2f} {r['R2']:>8.4f}{crown}")
    print(f"\n✅ Best model: {results[0]['Model']} (MAE=${results[0]['MAE']:,.2f})")

if __name__ == "__main__":
    run()

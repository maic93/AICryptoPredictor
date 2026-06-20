"""
Day 10 — Random Forest + Feature Importance
Week 2: Classical ML
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
    return df[feat_cols].values, df["target"].values, feat_cols

def run():
    print("🌲 Day 10 — Random Forest + Feature Importance")
    print("=" * 50)
    X, y, feat_cols = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  R²   : {r2:.4f}")

    print("\n  📊 Top 5 Feature Importances:")
    importances = sorted(zip(feat_cols, model.feature_importances_), key=lambda x: -x[1])
    for name, score in importances[:5]:
        bar = "█" * int(score * 100)
        print(f"    {name:>15s}: {bar} ({score:.4f})")
    print("\n✅ Random Forest complete!")

if __name__ == "__main__":
    run()

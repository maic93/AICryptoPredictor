"""
Day 09 — Decision Tree Regressor
Week 2: Classical ML
"""
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
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
    return df[feat_cols].values, df["target"].values

def run():
    print("🌳 Day 09 — Decision Tree Regressor")
    print("=" * 50)
    X, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    results = []
    for depth in [3, 5, 10, None]:
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        label = str(depth) if depth else "None"
        results.append((label, mae, rmse, r2))
        print(f"  depth={label:>4s} | MAE=${mae:>9,.2f} | RMSE=${rmse:>9,.2f} | R²={r2:.4f}")

    best = min(results, key=lambda x: x[1])
    print(f"\n  🏆 Best depth: {best[0]} (MAE=${best[1]:,.2f})")
    print("\n✅ Decision Tree complete!")

if __name__ == "__main__":
    run()

"""
Day 41 — Sentiment-Enhanced Price Prediction Model
Week 6: Sentiment Analysis
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os

os.makedirs("reports", exist_ok=True)

def simulate_sentiment(n: int) -> np.ndarray:
    np.random.seed(42)
    s = np.cumsum(np.random.normal(0, 0.08, n))
    s = np.clip(s / abs(s).max(), -1, 1)
    s += 0.3 * np.sin(np.linspace(0, 4*np.pi, n))
    return np.clip(s, -1, 1)

def build_dataset(ticker="BTC-USD"):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.dropna(inplace=True)
    close = df["Close"].squeeze()
    for lag in [1, 2, 3, 5, 7, 14]:
        df[f"lag{lag}"] = close.shift(lag)
    for w in [7, 14, 30]:
        df[f"rollMean{w}"] = close.rolling(w).mean()
        df[f"rollStd{w}"]  = close.rolling(w).std()
    df["return1d"] = close.pct_change()
    df["return7d"]  = close.pct_change(7)
    df["target"]   = close.shift(-1)
    df.dropna(inplace=True)
    sentiment = simulate_sentiment(len(df))
    df["sentiment"]     = sentiment
    df["sentiment_ma7"] = pd.Series(sentiment).rolling(7, min_periods=1).mean().values
    feat_base = [c for c in df.columns if c not in
                 ["target","Open","High","Low","Close","Volume","sentiment","sentiment_ma7"]]
    feat_sent = feat_base + ["sentiment", "sentiment_ma7"]
    feat_base = [str(c) for c in feat_base]
    feat_sent = [str(c) for c in feat_sent]
    df.columns = [str(c) for c in df.columns]
    return df[feat_base], df[feat_sent], df["target"], df.index

def run():
    print("Day 41 - Sentiment-Enhanced Prediction Model")
    print("=" * 55)
    X_base, X_sent, y, idx = build_dataset()
    split = int(len(X_base) * 0.8)
    Xb_tr, Xb_te = X_base.iloc[:split], X_base.iloc[split:]
    Xs_tr, Xs_te = X_sent.iloc[:split], X_sent.iloc[split:]
    y_tr,  y_te  = y.iloc[:split],       y.iloc[split:]
    dates_te      = idx[split:]

    model_base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_sent = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_base.fit(Xb_tr, y_tr)
    model_sent.fit(Xs_tr, y_tr)

    preds_base = model_base.predict(Xb_te)
    preds_sent = model_sent.predict(Xs_te)

    mae_base = mean_absolute_error(y_te, preds_base)
    mae_sent = mean_absolute_error(y_te, preds_sent)
    r2_base  = r2_score(y_te, preds_base)
    r2_sent  = r2_score(y_te, preds_sent)

    improvement = (mae_base - mae_sent) / mae_base * 100
    print(f"  Without sentiment: MAE=${mae_base:,.2f}  R2={r2_base:.4f}")
    print(f"  With sentiment   : MAE=${mae_sent:,.2f}  R2={r2_sent:.4f}")
    print(f"  MAE improvement  : {improvement:+.1f}%")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    axes[0].plot(dates_te, y_te.values,   color="#F7931A", linewidth=1.5, label="Actual")
    axes[0].plot(dates_te, preds_base, color="#3498DB", linewidth=1.2,
                 linestyle="--", label=f"Without Sentiment (MAE=${mae_base:,.0f})")
    axes[0].plot(dates_te, preds_sent, color="#2ECC71", linewidth=1.2,
                 linestyle=":",  label=f"With Sentiment (MAE=${mae_sent:,.0f})")
    axes[0].set_title("Sentiment-Enhanced Prediction vs Baseline", fontweight="bold")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    models  = ["Without Sentiment", "With Sentiment"]
    maes    = [mae_base, mae_sent]
    bcolors = ["#3498DB", "#2ECC71"]
    bars = axes[1].bar(models, maes, color=bcolors, alpha=0.85, width=0.4)
    axes[1].set_title("MAE Comparison (lower = better)", fontweight="bold")
    axes[1].set_ylabel("MAE (USD)"); axes[1].grid(alpha=0.3, axis="y")
    for bar, mae in zip(bars, maes):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                     f"${mae:,.0f}", ha="center", fontsize=11, fontweight="bold")
    axes[1].text(0.5, 0.85, f"Improvement: {improvement:+.1f}%",
                  ha="center", transform=axes[1].transAxes,
                  fontsize=13, fontweight="bold",
                  color="#2ECC71" if improvement > 0 else "#E74C3C")

    plt.suptitle("Day 41 - Sentiment-Enhanced Price Prediction",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day41chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")
    print("\n✅ Day 41 complete!")

if __name__ == "__main__":
    run()

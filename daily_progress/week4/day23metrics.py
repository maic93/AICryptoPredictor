"""
Day 23 — Sharpe Ratio, Drawdown & Strategy Metrics
Week 4: Production
"""
import numpy as np
import pandas as pd
import yfinance as yf

def load_returns(ticker="BTC-USD"):
    df = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    close = df["Close"].squeeze().dropna()
    return close.pct_change().dropna()

def sharpe(returns: pd.Series, risk_free=0.05) -> float:
    daily_rf = risk_free / 252
    excess = returns - daily_rf
    return float(np.sqrt(252) * excess.mean() / excess.std())

def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())

def calmar(returns: pd.Series) -> float:
    annual_return = float((1 + returns.mean()) ** 252 - 1)
    dd = abs(max_drawdown(returns))
    return annual_return / dd if dd != 0 else 0

def run():
    print("📐 Day 23 — Strategy Metrics")
    print("=" * 50)
    for coin in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        r = load_returns(coin)
        ann_ret = float((1 + r.mean()) ** 252 - 1) * 100
        vol     = float(r.std() * np.sqrt(252)) * 100
        sr      = sharpe(r)
        md      = max_drawdown(r) * 100
        cm      = calmar(r)
        print(f"\n  [{coin}]")
        print(f"    Annual Return : {ann_ret:+.2f}%")
        print(f"    Volatility    : {vol:.2f}%")
        print(f"    Sharpe Ratio  : {sr:.3f}")
        print(f"    Max Drawdown  : {md:.2f}%")
        print(f"    Calmar Ratio  : {cm:.3f}")
    print("\n✅ Day 23 complete!")

if __name__ == "__main__":
    run()

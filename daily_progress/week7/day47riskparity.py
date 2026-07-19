"""
Day 47 — Risk-Parity Portfolio Allocation
Week 7: Portfolio Optimizer
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

os.makedirs("reports", exist_ok=True)
COINS  = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"]
LABELS = ["BTC","ETH","SOL","BNB"]
COLORS = ["#F7931A","#627EEA","#9945FF","#F0B90B"]

def fetch_returns():
    all_close = {}
    for coin in COINS:
        df = yf.download(coin, period="1y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        all_close[coin.replace("-USD","")] = df["Close"].squeeze()
    return pd.DataFrame(all_close).dropna().pct_change().dropna()

def risk_contribution(weights, cov):
    port_vol = np.sqrt(weights @ cov @ weights)
    marginal = cov @ weights
    return weights * marginal / port_vol

def risk_parity_objective(weights, cov):
    rc   = risk_contribution(weights, cov)
    target = np.ones(len(weights)) / len(weights)
    return float(np.sum((rc - target)**2))

def optimize_rp(cov):
    n    = len(COINS)
    w0   = np.ones(n) / n
    bounds      = [(0.01, 0.99)] * n
    constraints = {"type":"eq","fun":lambda w: w.sum()-1}
    result = minimize(risk_parity_objective, w0, args=(cov,),
                       method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x

def run():
    print("Day 47 - Risk-Parity Portfolio Allocation")
    print("=" * 55)
    print("  Risk-parity: each asset contributes equally to portfolio risk.\n")
    returns  = fetch_returns()
    cov      = returns.cov().values
    vols     = returns.std().values * np.sqrt(252) * 100

    rp_weights = optimize_rp(cov)
    eq_weights = np.ones(len(COINS)) / len(COINS)
    iv_weights = (1/vols) / (1/vols).sum()  # inverse-vol weights

    rc_rp = risk_contribution(rp_weights, cov)
    rc_eq = risk_contribution(eq_weights, cov)
    rc_iv = risk_contribution(iv_weights, cov)

    print(f"  {'Coin':<8} {'Vol':>8} {'EqW%':>8} {'InvVol%':>10} {'RiskPar%':>10}")
    print("  " + "-" * 50)
    for i, label in enumerate(LABELS):
        print(f"  {label:<8} {vols[i]:>7.1f}% {eq_weights[i]*100:>7.1f}%"
              f" {iv_weights[i]*100:>9.1f}% {rp_weights[i]*100:>9.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, weights, title in zip(
        axes,
        [eq_weights, iv_weights, rp_weights],
        ["Equal Weight", "Inverse-Vol", "Risk-Parity"]
    ):
        ax.pie(weights, labels=LABELS, colors=COLORS,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor":"white","linewidth":2})
        ax.set_title(title, fontweight="bold")

    plt.suptitle("Day 47 - Risk-Parity vs Equal vs Inverse-Vol Allocation",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = "reports/day47chart.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Chart saved -> {out}")
    print("\n✅ Day 47 complete!")

if __name__ == "__main__":
    run()

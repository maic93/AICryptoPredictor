"""
Day 50 — Streamlit App Structure & Layout
Week 8: Web Dashboard
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("reports", exist_ok=True)
os.makedirs("dashboard", exist_ok=True)

STREAMLIT_APP = '''"""
AICryptoPredictor — Streamlit Dashboard
Run locally: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="AICryptoPredictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("AICryptoPredictor")
st.sidebar.markdown("**AI-powered crypto analysis**")
selected_coin = st.sidebar.selectbox(
    "Select Coin",
    ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"],
)
period = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y","2y"])

# Main layout
st.title("AICryptoPredictor Dashboard")
st.markdown(f"*Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*")

col1, col2, col3, col4 = st.columns(4)

@st.cache_data(ttl=300)
def load_data(ticker, period):
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if hasattr(df.columns, "levels"):
        df.columns = [c[0] for c in df.columns]
    return df.dropna()

df = load_data(selected_coin, period)
close = df["Close"].squeeze()

price     = float(close.iloc[-1])
price_1d  = float(close.iloc[-2]) if len(close) > 1 else price
change_1d = (price - price_1d) / price_1d * 100

col1.metric("Price", f"${price:,.2f}", f"{change_1d:+.2f}%")
col2.metric("High (period)", f"${float(close.max()):,.2f}")
col3.metric("Low (period)",  f"${float(close.min()):,.2f}")
col4.metric("Vol (daily)",
            f"{float(close.pct_change().std()*100):.2f}%")

# Price chart
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"].squeeze(),
    high=df["High"].squeeze(),
    low=df["Low"].squeeze(),
    close=close,
    name=selected_coin,
))
fig.update_layout(title=f"{selected_coin} Price Chart",
                   template="plotly_dark", height=400,
                   xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("*Educational only — not financial advice*")
'''

def plot_layout_preview():
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#262730")
        for sp in ax.spines.values():
            sp.set_color("#444")

    # Mock header
    axes[0].text(0.02, 0.8, "AICryptoPredictor Dashboard", fontsize=16,
                  color="white", fontweight="bold", transform=axes[0].transAxes)
    axes[0].text(0.02, 0.4, "Sidebar: Coin selector | Period | Settings",
                  fontsize=11, color="#aaa", transform=axes[0].transAxes)
    mock_metrics = [("BTC Price","$65,420","+ 2.3%"), ("Period High","$72,000",""),
                     ("Period Low","$55,000",""), ("Daily Vol","3.2%","")]
    for i, (label, val, delta) in enumerate(mock_metrics):
        x = 0.02 + i * 0.25
        axes[0].text(x, 0.05, f"{label}\n{val} {delta}",
                      fontsize=9, color="white", transform=axes[0].transAxes,
                      bbox=dict(boxstyle="round", facecolor="#1c1c2e", alpha=0.8))
    axes[0].set_title("Page Header + Metric Cards", color="white", fontweight="bold")
    axes[0].axis("off")

    # Mock candlestick
    np.random.seed(42)
    n     = 30
    price = 65000 + np.cumsum(np.random.randn(n) * 500)
    axes[1].plot(range(n), price, color="#F7931A", linewidth=2)
    axes[1].fill_between(range(n), price*0.99, price*1.01, alpha=0.2, color="#F7931A")
    axes[1].set_title("Price Chart Component (Plotly Candlestick)", color="white", fontweight="bold")
    axes[1].tick_params(colors="white"); axes[1].grid(alpha=0.2)

    # Mock component grid
    axes[2].text(0.15, 0.5, "Prediction\nChart", ha="center", va="center",
                  fontsize=12, color="white", transform=axes[2].transAxes,
                  bbox=dict(facecolor="#1c1c2e", alpha=0.8, boxstyle="round"))
    axes[2].text(0.40, 0.5, "Signal\nDashboard", ha="center", va="center",
                  fontsize=12, color="white", transform=axes[2].transAxes,
                  bbox=dict(facecolor="#1c1c2e", alpha=0.8, boxstyle="round"))
    axes[2].text(0.65, 0.5, "Sentiment\nGauge", ha="center", va="center",
                  fontsize=12, color="white", transform=axes[2].transAxes,
                  bbox=dict(facecolor="#1c1c2e", alpha=0.8, boxstyle="round"))
    axes[2].text(0.88, 0.5, "Portfolio\nPie", ha="center", va="center",
                  fontsize=12, color="white", transform=axes[2].transAxes,
                  bbox=dict(facecolor="#1c1c2e", alpha=0.8, boxstyle="round"))
    axes[2].set_title("Component Grid (Days 51-55)", color="white", fontweight="bold")
    axes[2].axis("off")

    plt.suptitle("Day 50 - Streamlit Dashboard Layout Preview",
                  fontsize=13, fontweight="bold", color="white")
    plt.tight_layout()
    out = "reports/day50chart.png"
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Chart saved -> {out}")

def run():
    print("Day 50 - Streamlit App Structure & Layout")
    print("=" * 55)

    with open("dashboard/app.py", "w") as f:
        f.write(STREAMLIT_APP)
    print("  Streamlit app created -> dashboard/app.py")
    print("\n  To run locally:")
    print("    pip install streamlit plotly")
    print("    streamlit run dashboard/app.py")
    print("\n  App features:")
    print("    - Coin selector sidebar (BTC/ETH/SOL/BNB)")
    print("    - Period selector (1mo to 2y)")
    print("    - Metric cards (price, high, low, volatility)")
    print("    - Interactive Plotly candlestick chart")
    print("    - Dark theme")
    print("\n  Components to be added Days 51-55:")
    print("    - Prediction chart")
    print("    - Signal dashboard")
    print("    - Sentiment gauge")
    print("    - Portfolio pie chart")

    plot_layout_preview()
    print("\n✅ Day 50 complete!")

if __name__ == "__main__":
    run()

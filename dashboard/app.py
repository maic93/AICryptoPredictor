"""
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

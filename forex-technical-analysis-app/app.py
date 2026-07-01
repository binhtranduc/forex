import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Forex AI Bot", layout="wide")

st.title("🤖 Forex AI Trading Bot (Stable Version)")

# ======================
# AUTO REFRESH (SAFE)
# ======================
st.markdown("""
<script>
setTimeout(function(){
    window.location.reload();
}, 30000);
</script>
""", unsafe_allow_html=True)

# ======================
# SYMBOLS
# ======================
DATA_SYMBOL_MAP = {
    "EURUSD=X": "EURUSD=X",
    "GBPUSD=X": "GBPUSD=X",
    "USDJPY=X": "USDJPY=X",
    "AUDUSD=X": "AUDUSD=X",
    "USDCAD=X": "USDCAD=X",
    "NZDUSD=X": "NZDUSD=X",
    "XAUUSD": "GC=F",
}

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    symbol = st.selectbox("Market", list(DATA_SYMBOL_MAP.keys()))
    period = st.selectbox("Period", ["1mo", "3mo", "6mo"], index=0)
    interval = st.selectbox("Interval", ["5m", "15m", "1h", "1d"], index=1)

# ======================
# LOAD DATA (FIXED SAFE)
# ======================
@st.cache_data
def load_data(symbol, period, interval):
    df = yf.download(DATA_SYMBOL_MAP[symbol],
                     period=period,
                     interval=interval)

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Indicators
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()

    df["MACD"] = ema12 - ema26
    df["SIGNAL"] = df["MACD"].ewm(span=9).mean()

    df = df.dropna()
    df = df.reset_index()

    return df

df = load_data(symbol, period, interval)

if df.empty:
    st.error("No data available")
    st.stop()

# ======================
# SAFE LAST VALUE (FIX ERROR YOU HAD)
# ======================
last_close = float(df["Close"].iloc[-1])

# ======================
# AI SIGNAL
# ======================
def ai_signal(df):
    last = df.iloc[-1]
    score = 0

    if last["RSI"] < 30:
        score += 2
    elif last["RSI"] > 70:
        score -= 2

    if last["SMA20"] > last["SMA50"]:
        score += 1
    else:
        score -= 1

    if last["MACD"] > last["SIGNAL"]:
        score += 1
    else:
        score -= 1

    if score >= 2:
        return "STRONG BUY"
    elif score <= -2:
        return "STRONG SELL"
    else:
        return "HOLD"

signal = ai_signal(df)

# ======================
# METRICS (FIXED ERROR HERE)
# ======================
col1, col2, col3 = st.columns(3)

col1.metric("Market", symbol)
col2.metric("Price", f"{last_close:.5f}")
col3.metric("Signal", signal)

# ======================
# CHART
# ======================
fig = make_subplots(rows=1, cols=1)

fig.add_trace(go.Candlestick(
    x=df["Datetime"] if "Datetime" in df.columns else df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
))

fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"))
fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"))

st.plotly_chart(fig, use_container_width=True)

# ======================
# MARKET BIAS
# ======================
if last_close > df["SMA20"].iloc[-1]:
    bias = "Bullish"
else:
    bias = "Bearish"

st.subheader("Market Bias")
st.write(bias)

# ======================
# TABLE
# ======================
st.subheader("Latest Data")
st.dataframe(df.tail(10))

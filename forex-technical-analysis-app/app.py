import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.parse import quote
import streamlit.components.v1 as components

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Forex AI Trading Bot", layout="wide")

st.title("🤖 Forex AI Trading Bot (Fixed Version)")
st.caption("Real-time indicators + AI signals + TradingView chart")

# ======================
# AUTO REFRESH (FIX)
# ======================
st_autorefresh = st.empty()
st_autorefresh.markdown("""
<script>
setTimeout(function(){
   window.location.reload();
}, 30000);
</script>
""", unsafe_allow_html=True)

# ======================
# SYMBOLS
# ======================
SYMBOL_MAP = {
    "EURUSD=X": "FX:EURUSD",
    "GBPUSD=X": "FX:GBPUSD",
    "USDJPY=X": "FX:USDJPY",
    "AUDUSD=X": "FX:AUDUSD",
    "USDCAD=X": "FX:USDCAD",
    "NZDUSD=X": "FX:NZDUSD",
    "XAUUSD": "OANDA:XAUUSD",
}

DATA_SYMBOL_MAP = {
    "EURUSD=X": "EURUSD=X",
    "GBPUSD=X": "GBPUSD=X",
    "USDJPY=X": "USDJPY=X",
    "AUDUSD=X": "AUDUSD=X",
    "USDCAD=X": "USDCAD=X",
    "NZDUSD=X": "NZDUSD=X",
    "XAUUSD": "GC=F",
}

INTERVAL_MAP = {
    "5m": "5",
    "15m": "15",
    "1h": "60",
    "1d": "1d",
    "1wk": "1wk",
}

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.header("Settings")

    symbol = st.selectbox("Market", list(SYMBOL_MAP.keys()))
    period = st.selectbox("Period", ["7d", "1mo", "3mo", "6mo", "1y"], index=1)
    interval = st.selectbox("Interval", ["5m", "15m", "1h", "1d"], index=1)

    st.subheader("Indicators")
    show_sma20 = st.checkbox("SMA 20", True)
    show_sma50 = st.checkbox("SMA 50", True)
    show_rsi = st.checkbox("RSI", True)
    show_macd = st.checkbox("MACD", True)
    show_bb = st.checkbox("Bollinger Bands", True)

    enable_ai = st.checkbox("Enable AI Signal", True)

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data(symbol, period, interval):

    df = yf.download(DATA_SYMBOL_MAP[symbol], period=period, interval=interval)

    if df.empty:
        return df

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

    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()

    df["BB_UP"] = mid + 2 * std
    df["BB_LOW"] = mid - 2 * std

    df = df.dropna()
    return df

df = load_data(symbol, period, interval)

if df.empty:
    st.error("No data")
    st.stop()

latest = df.iloc[-1]

# ======================
# AI SIGNAL (FIXED SAFE)
# ======================
def ai_signal(df):
    last = df.iloc[-1]

    score = 0

    rsi = last["RSI"]

    if rsi < 30:
        score += 2
    elif rsi > 70:
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
        return "STRONG BUY", score
    elif score <= -2:
        return "STRONG SELL", score
    else:
        return "HOLD", score

# ======================
# TOP METRICS
# ======================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Market", symbol)
col2.metric("Price", f"{latest['Close']:.5f}")
col3.metric("RSI", f"{latest['RSI']:.2f}")

signal, score = ai_signal(df)
col4.metric("Signal", signal)

# ======================
# AI DISPLAY
# ======================
if enable_ai:
    if signal == "STRONG BUY":
        st.success(f"🟢 {signal}")
    elif signal == "STRONG SELL":
        st.error(f"🔴 {signal}")
    else:
        st.warning(f"🟡 {signal}")

# ======================
# CHART
# ======================
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
), row=1, col=1)

if show_sma20:
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"), row=1, col=1)

if show_sma50:
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"), row=1, col=1)

if show_bb:
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"], name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOW"], name="BB Lower"), row=1, col=1)

fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)

if show_rsi:
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ======================
# BIAS
# ======================
if latest["Close"] > latest["SMA20"] and latest["RSI"] > 50:
    bias = "Bullish"
elif latest["Close"] < latest["SMA20"] and latest["RSI"] < 50:
    bias = "Bearish"
else:
    bias = "Neutral"

st.subheader("Market Bias")
st.write(bias)

# ======================
# TRADINGVIEW
# ======================
tv_symbol = quote(SYMBOL_MAP[symbol])

url = f"https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval={INTERVAL_MAP[interval]}&theme=dark"

st.subheader("TradingView Chart")
st.components.v1.iframe(url, height=600)

# ======================
# TABLE
# ======================
st.subheader("Latest Data")
st.dataframe(df.tail(10))

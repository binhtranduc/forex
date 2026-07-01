import sysimport pandas as pdimport numpy as npimport streamlit as stimport yfinance as yfimport plotly.graph_objects as gofrom plotly.subplots import make_subplotsfrom urllib.parse import quoteimport time

st.set_page_config(page_title="Forex Technical Analysis", page_icon="📈", layout="wide")

st.title("Forex Technical Analysis Dashboard")st.write("Real-time forex and gold charts from TradingView plus AI trading signals.")

SYMBOL_MAP = {"EURUSD=X": "FX","GBPUSD=X": "FX","USDJPY=X": "FX","AUDUSD=X": "FX","USDCAD=X": "FX","NZDUSD=X": "FX","XAUUSD": "OANDA",}

DATA_SYMBOL_MAP = {"EURUSD=X": "EURUSD=X","GBPUSD=X": "GBPUSD=X","USDJPY=X": "USDJPY=X","AUDUSD=X": "AUDUSD=X","USDCAD=X": "USDCAD=X","NZDUSD=X": "NZDUSD=X","XAUUSD": "GC=F",}

INTERVAL_MAP = {"5m": "5","15m": "15","1h": "60","1d": "D","1wk": "W",}

with st.sidebar:st.header("Settings")symbol = st.selectbox("Market",list(SYMBOL_MAP.keys()),index=0,)period = st.selectbox("Period", ["7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=1)interval = st.selectbox("Interval", ["5m", "15m", "1h", "1d", "1wk"], index=1)

st.subheader("Indicators")
show_sma20 = st.checkbox("SMA 20", value=True)
show_sma50 = st.checkbox("SMA 50", value=True)
show_rsi = st.checkbox("RSI 14", value=True)
show_macd = st.checkbox("MACD", value=True)
show_bbands = st.checkbox("Bollinger Bands", value=True)

st.subheader("AI Analysis")
enable_ai = st.checkbox("Enable AI Trading Signal", value=True)

@st.cache_data(show_spinner=False)def load_data(symbol: str, period: str, interval: str) -> pd.DataFrame:yfinance_symbol = DATA_SYMBOL_MAP[symbol]

if interval == "5m" and period not in {"7d", "1mo"}:
    period = "7d"
elif interval == "15m" and period in {"6mo", "1y", "2y", "5y", "max"}:
    period = "2mo"
elif interval == "1h" and period in {"2y", "5y", "max"}:
    period = "1y"

data = yf.download(yfinance_symbol, period=period, interval=interval, auto_adjust=False, progress=False)
if data.empty or (isinstance(data.columns, pd.MultiIndex) and "Close" not in data.columns.get_level_values(0)):
    fallback_interval = "1d"
    fallback_period = "1y"
    data = yf.download(yfinance_symbol, period=fallback_period, interval=fallback_interval, auto_adjust=False, progress=False)

if data.empty:
    raise ValueError(f"No data returned for {symbol}.")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(-1)

if "Close" not in data.columns:
    raise ValueError(f"Unexpected data format returned for {symbol}.")

data = data.copy()
data.index = pd.to_datetime(data.index)
data = data.sort_index()

data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["SMA_50"] = data["Close"].rolling(window=50).mean()

delta = data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
data["RSI_14"] = 100 - (100 / (1 + rs))

ema12 = data["Close"].ewm(span=12, adjust=False).mean()
ema26 = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"] = ema12 - ema26
data["MACD_SIGNAL"] = data["MACD"].ewm(span=9, adjust=False).mean()
data["MACD_HIST"] = data["MACD"] - data["MACD_SIGNAL"]

mid = data["Close"].rolling(window=20).mean()
std = data["Close"].rolling(window=20).std()
data["BB_UPPER"] = mid + 2 * std
data["BB_MIDDLE"] = mid
data["BB_LOWER"] = mid - 2 * std

data = data.reset_index()
if "Date" not in data.columns and "index" in data.columns:
    data = data.rename(columns={"index": "Date"})
if "Date" not in data.columns:
    data["Date"] = data.index
return data

def predict_next_price(df: pd.DataFrame, lookback: int = 12) -> tuple[float, str, float]:"""Predict the next close price using a simple linear regression on recent closes."""closes = df["Close"].dropna()if len(closes) < 5:return float(closes.iloc[-1]), "Insufficient history", 0.0recent = closes.tail(lookback)x = np.arange(len(recent), dtype=float)y = recent.valuesm, b = np.polyfit(x, y, 1)trend_pred = float(m * len(x) + b)

# Autoregressive model using the last 3 bars
ar_pred = trend_pred
if len(recent) >= 4:
    X = []
    Y = []
    for i in range(3, len(recent)):
        X.append([recent.iloc[i - 1], recent.iloc[i - 2], recent.iloc[i - 3], 1.0])
        Y.append(recent.iloc[i])
    X = np.array(X)
    Y = np.array(Y)
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    ar_pred = float(
        coeffs[0] * recent.iloc[-1]
        + coeffs[1] * recent.iloc[-2]
        + coeffs[2] * recent.iloc[-3]
        + coeffs[3]
    )

# Momentum bias from EMA crossover
ema5 = recent.ewm(span=5, adjust=False).mean().iloc[-1]
ema10 = recent.ewm(span=10, adjust=False).mean().iloc[-1]
momentum_bias = (ema5 - ema10) * 0.06

predicted = float(0.55 * trend_pred + 0.35 * ar_pred + momentum_bias)
last_price = float(y[-1])
change_pct = (predicted - last_price) / last_price * 100 if last_price else 0.0
direction = "Up" if predicted > last_price else "Down" if predicted < last_price else "Flat"
return predicted, direction, change_pct

def get_ai_signal(df: pd.DataFrame) -> tuple:"""Advanced AI trading signal with weighted indicators"""latest = df.iloc[-1].to_dict()prev = df.iloc[-2].to_dict() if len(df) > 1 else latest

latest_close = float(latest["Close"])
prev_close = float(prev["Close"])

# Initialize weighted scoring system
score = 0  # -100 (strong sell) to +100 (strong buy)
reasons = []

# 1. RSI Analysis (Weight: 15%)
rsi = latest.get("RSI_14", 50)
if not pd.isna(rsi):
    if rsi < 20:
        score += 20
        reasons.append("🔥 RSI extremely oversold (< 20) - STRONG BUY")
    elif rsi < 30:
        score += 12
        reasons.append("✅ RSI oversold (< 30) - BUY signal")
    elif rsi > 80:
        score -= 20
        reasons.append("🔥 RSI extremely overbought (> 80) - STRONG SELL")
    elif rsi > 70:
        score -= 12
        reasons.append("⚠️ RSI overbought (> 70) - SELL signal")
    elif 45 < rsi < 55:
        score += 0
        reasons.append("➖ RSI neutral (45-55)")
    else:
        score += (rsi - 50) * 0.2  # Slight bias based on RSI position

# 2. Moving Average Convergence (Weight: 20%)
sma20 = latest.get("SMA_20", None)
sma50 = latest.get("SMA_50", None)

if not pd.isna(sma20) and not pd.isna(sma50):
    ma_ratio = (sma20 - sma50) / sma50 * 100
    if sma20 > sma50:
        if ma_ratio > 2:
            score += 18
            reasons.append("🔥 Strong uptrend (SMA20 >> SMA50)")
        else:
            score += 10
            reasons.append("✅ Uptrend confirmed (SMA20 > SMA50)")
    else:
        if ma_ratio < -2:
            score -= 18
            reasons.append("🔥 Strong downtrend (SMA20 << SMA50)")
        else:
            score -= 10
            reasons.append("⚠️ Downtrend confirmed (SMA20 < SMA50)")

# 3. Price Position Analysis (Weight: 15%)
if not pd.isna(sma20):
    price_sma_pct = (latest_close - sma20) / sma20 * 100
    if latest_close > sma20 * 1.02:
        score += 8
        reasons.append("✅ Price >> SMA20 (strong momentum)")
    elif latest_close > sma20:
        score += 4
        reasons.append("✅ Price above SMA20")
    elif latest_close < sma20 * 0.98:
        score -= 8
        reasons.append("⚠️ Price << SMA20 (weak momentum)")
    else:
        score -= 4
        reasons.append("⚠️ Price below SMA20")

# 4. Bollinger Bands Squeeze & Reversal (Weight: 15%)
bb_upper = latest.get("BB_UPPER", None)
bb_lower = latest.get("BB_LOWER", None)

if not pd.isna(bb_upper) and not pd.isna(bb_lower):
    bb_width = bb_upper - bb_lower
    bb_position = (latest_close - bb_lower) / bb_width if bb_width > 0 else 0.5
    
    if latest_close > bb_upper * 1.01:
        score -= 15
        reasons.append("🔥 Price far above upper BB - EXTREME overbought")
    elif latest_close > bb_upper:
        score -= 8
        reasons.append("⚠️ Price above upper BB - Overbought")
    elif latest_close < bb_lower * 0.99:
        score += 15
        reasons.append("🔥 Price far below lower BB - EXTREME oversold")
    elif latest_close < bb_lower:
        score += 8
        reasons.append("✅ Price below lower BB - Oversold bounce")
    elif 0.4 < bb_position < 0.6:
        score += 3
        reasons.append("✅ BB squeeze - Breakout expected")

# 5. MACD Momentum (Weight: 15%)
macd = latest.get("MACD", None)
macd_signal = latest.get("MACD_SIGNAL", None)
macd_hist = latest.get("MACD_HIST", None)

if not pd.isna(macd) and not pd.isna(macd_signal):
    macd_prev = df.iloc[-2].get("MACD", macd) if len(df) > 1 else macd
    macd_signal_prev = df.iloc[-2].get("MACD_SIGNAL", macd_signal) if len(df) > 1 else macd_signal
    
    # Check for crossover
    if macd_prev <= macd_signal_prev and macd > macd_signal:
        score += 16
        reasons.append("🔥 MACD bullish crossover - Strong momentum shift")
    elif macd_prev >= macd_signal_prev and macd < macd_signal:
        score -= 16
        reasons.append("🔥 MACD bearish crossover - Strong momentum shift")
    elif macd > macd_signal:
        score += 8
        reasons.append("✅ MACD above signal (bullish)")
    else:
        score -= 8
        reasons.append("⚠️ MACD below signal (bearish)")

# 6. Momentum Direction (Weight: 10%)
if prev_close != 0:
    price_momentum = (latest_close - prev_close) / prev_close * 100
    if price_momentum > 0.5:
        score += 5
        reasons.append("✅ Positive momentum")
    elif price_momentum < -0.5:
        score -= 5
        reasons.append("⚠️ Negative momentum")

# 7. Volatility Context (Weight: 10%)
bb_upper = latest.get("BB_UPPER", None)
bb_lower = latest.get("BB_LOWER", None)
if not pd.isna(bb_upper) and not pd.isna(bb_lower):
    bb_range = bb_upper - bb_lower
    if bb_range < 0.5:
        reasons.append("⏰ Low volatility - Breakout likely soon")
    elif bb_range > 3:
        reasons.append("⚡ High volatility - Large moves expected")

# Determine final signal with confidence
if score >= 40:
    signal = "🟢 STRONG BUY"
    confidence = "VERY HIGH"
elif score >= 20:
    signal = "🟢 BUY"
    confidence = "HIGH"
elif score >= 5:
    signal = "🟢 BUY"
    confidence = "MEDIUM"
elif score <= -40:
    signal = "🔴 STRONG SELL"
    confidence = "VERY HIGH"
elif score <= -20:
    signal = "🔴 SELL"
    confidence = "HIGH"
elif score <= -5:
    signal = "🔴 SELL"
    confidence = "MEDIUM"
else:
    signal = "🟡 HOLD"
    confidence = "MEDIUM"

# Determine trading status
if "STRONG" in signal:
    status = "🎯 GET IN"
elif confidence == "VERY HIGH":
    status = "🎯 GET IN"
elif confidence == "HIGH":
    status = "✅ READY"
elif confidence == "MEDIUM" and ("BUY" in signal or "SELL" in signal):
    status = "✅ READY"
else:
    status = "⏳ WAIT"

# Risk/Reward assessment
rsi = latest.get("RSI_14", 50)
if score > 0:
    if rsi > 60:
        risk = "Medium Risk: Price momentum strong but RSI elevated"
    else:
        risk = "Low Risk: Good entry with confirmed momentum"
elif score < 0:
    if rsi < 40:
        risk = "Medium Risk: Price weak but RSI not extremely low"
    else:
        risk = "Low Risk: Clear downtrend with high probability"
else:
    risk = "Unclear Risk: Wait for clearer signal"

reasons.append(f"📊 Score: {score}/100 - {risk}")
explanation = " | ".join(reasons)

return signal, confidence, status, explanation

try:df = load_data(symbol, period, interval)except Exception as exc:st.error(str(exc))st.stop()

latest = df.iloc[-1].to_dict()previous = df.iloc[-2].to_dict() if len(df) > 1 else latestlatest_close = float(latest["Close"])previous_close = float(previous["Close"])price_change = latest_close - previous_closeprice_change_pct = (price_change / previous_close) * 100 if previous_close else 0.0

col1, col2, col3, col4 = st.columns(4)col1.metric("Market", symbol)col2.metric("Last Close", f"{latest_close:.5f}")col3.metric("Change", f"{price_change:.5f} ({price_change_pct:.2f}%)")col4.metric("RSI (14)", f"{latest['RSI_14']:.2f}")

if enable_ai and interval == "15m":st.subheader("🤖 Advanced AI Trading Signal (15m)")signal, confidence, status, explanation = get_ai_signal(df)

# 🔔 GET IN Notification
if "GET IN" in status:
    st.balloons()
    
    # Sound alert and pulsing effect
    st.html("""
    <style>
        @keyframes pulse-alert {
            0%, 100% { background-color: #ff4444; }
            50% { background-color: #ff0000; }
        }
        .alert-pulse {
            animation: pulse-alert 0.5s infinite;
            padding: 15px;
            border-radius: 5px;
            color: white;
            font-weight: bold;
            text-align: center;
            font-size: 18px;
            margin: 10px 0;
        }
    </style>
    <div class="alert-pulse">🚨 🚨 🚨 ALERT! GET IN NOW! 🚨 🚨 🚨</div>
    <audio autoplay>
        <source src="data:audio/wav;base64,UklGRiYAAABXQVZFZm10IBAAAAABAAEAQB8AAAB9AAACABAAZGF0YQIAAAAAAA==" type="audio/wav">
    </audio>
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.error(f"Signal: {signal}")
    col2.error(f"Confidence: {confidence}")
    col3.error(f"Status: {status}")

# Display signal prominently
if "STRONG BUY" in signal:
    st.success(f"### {signal}")
elif "STRONG SELL" in signal:
    st.error(f"### {signal}")
elif "BUY" in signal:
    st.success(f"### {signal}")
elif "SELL" in signal:
    st.error(f"### {signal}")
else:
    st.warning(f"### {signal}")

# Detailed metrics with trading status
col1, col2, col3 = st.columns(3)
col1.metric("Signal Type", signal.split()[1])
col2.metric("Confidence Level", confidence)

# Status with color coding
if "GET IN" in status:
    col3.success(status)
elif "READY" in status:
    col3.info(status)
else:
    col3.warning(status)

# Detailed analysis with better formatting
st.subheader("📊 Detailed Analysis")
analysis_parts = explanation.split(" | ")

cols = st.columns(2)
for i, part in enumerate(analysis_parts):
    cols[i % 2].info(part)

elif enable_ai and interval != "15m":st.warning("⏰ AI analysis is optimized for 15m interval. Switch to 15m for trading signals.")

st.subheader("Live TradingView Chart")st.caption("This embedded widget loads the live market chart for the selected pair or gold.")

if interval == "5m":try:next_price, next_direction, next_change_pct = predict_next_price(df, lookback=15)st.subheader("🔮 5-Minute Price Prediction")st.metric("Predicted Close in 5m",f"{next_price:.5f}",f"{next_direction} {next_change_pct:+.2f}%",)st.caption("Prediction based on linear regression of the latest 5-minute bars.")except Exception as exc:st.warning(f"Unable to calculate 5-minute prediction: {exc}")

tradingview_symbol = quote(SYMBOL_MAP[symbol], safe="")tradingview_url = ("https://s.tradingview.com/widgetembed/?frameElementId=tradingview_chart"f"&symbol={tradingview_symbol}"f"&interval={INTERVAL_MAP[interval]}""&theme=dark""&style=1""&locale=en""&toolbarbg=f1f3f6""&allow_symbol_change=true""&saveimage=false")st.iframe(tradingview_url, height=720)st.link_button("Open this chart in TradingView", tradingview_url)

st.subheader("Technical Analysis")fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.03,row_heights=[0.75, 0.25],)

fig.add_trace(go.Candlestick(x=df["Date"],open=df["Open"],high=df["High"],low=df["Low"],close=df["Close"],name="Price",increasing_line_color="#2ecc71",decreasing_line_color="#e74c3c",),row=1,col=1,)

if show_sma20:fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"], mode="lines", name="SMA 20"), row=1, col=1)if show_sma50:fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_50"], mode="lines", name="SMA 50"), row=1, col=1)if show_bbands:fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_UPPER"], mode="lines", name="BB Upper", line=dict(dash="dot")), row=1, col=1)fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_LOWER"], mode="lines", name="BB Lower", line=dict(dash="dot")), row=1, col=1)

fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color="#7f8c8d"), row=2, col=1)

if show_rsi:fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI_14"], mode="lines", name="RSI 14"), row=2, col=1)fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(height=700, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))st.plotly_chart(fig, width="stretch")

st.subheader("Signal Summary")if latest_close > latest["SMA_20"] and latest["RSI_14"] > 50:signal_bias = "Bullish"elif latest_close < latest["SMA_20"] and latest["RSI_14"] < 50:signal_bias = "Bearish"else:signal_bias = "Neutral"

st.write(f"Current bias: {signal_bias}")

st.subheader("Latest Rows")latest_rows = df.tail(10).copy()latest_rows = latest_rows[["Date", "Open", "High", "Low", "Close", "Volume", "SMA_20", "SMA_50", "RSI_14"]]st.dataframe(latest_rows, width="stretch")

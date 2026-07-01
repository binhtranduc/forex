# Forex Technical Analysis Dashboard with AI

A real-time trading analysis app with TradingView charts and AI-powered trading signals.

## Features

- **Real-time Charts**: Live forex pairs and gold prices from TradingView
- **Technical Indicators**: SMA, RSI, MACD, Bollinger Bands
- **AI Trading Signals**: OpenAI GPT-powered buy/sell/hold recommendations for 15m timeframe
- **Multiple Markets**: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, NZD/USD, Gold

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get your OpenAI API key from [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. In the sidebar, paste your OpenAI API key and enable "Enable AI Trading Signal"

## Command-line prediction without browser

If you want to get a quick predicted next close price from the terminal without opening the Streamlit dashboard, run:
```bash
python predict_price.py --symbol EURUSD=X --interval 5m
```
This will print the current last close and the predicted next close for the following 5-minute bar.

## How to Use

1. Select a market (forex pair or gold) from the dropdown
2. Choose your desired timeframe (5m, 15m, 1h, 1d, 1wk)
3. Switch to **5m interval** to get a 5-minute price prediction, or **15m interval** to see AI trading signals
4. Enter your OpenAI API key in the sidebar
5. The AI will analyze technical indicators and provide:
   - **SIGNAL**: BUY, SELL, or HOLD
   - **CONFIDENCE**: High, Medium, or Low
   - **REASON**: Explanation based on technical analysis
   - **RISK**: Risk considerations

## Technical Indicators

- **SMA 20 & 50**: Moving averages for trend direction
- **RSI 14**: Momentum and overbought/oversold levels
- **MACD**: Trend and momentum confirmation
- **Bollinger Bands**: Volatility and support/resistance levels

## Important Notes

⚠️ This tool is for educational purposes. Always do your own research and never trade solely on AI signals.

## Costs

Using OpenAI API will incur charges based on token usage. Check your usage at https://platform.openai.com/usage

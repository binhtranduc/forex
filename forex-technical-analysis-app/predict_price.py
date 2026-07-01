import argparse
import numpy as np
import pandas as pd
import yfinance as yf

DATA_SYMBOL_MAP = {
    "EURUSD=X": "EURUSD=X",
    "GBPUSD=X": "GBPUSD=X",
    "USDJPY=X": "USDJPY=X",
    "AUDUSD=X": "AUDUSD=X",
    "USDCAD=X": "USDCAD=X",
    "NZDUSD=X": "NZDUSD=X",
    "XAUUSD": "GC=F",
}


def load_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    yfinance_symbol = DATA_SYMBOL_MAP[symbol]
    if interval == "5m" and period not in {"7d", "1mo"}:
        period = "7d"
    data = yf.download(yfinance_symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {symbol} using {interval} interval.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(-1)
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data


def predict_next_price(df: pd.DataFrame, lookback: int = 12) -> tuple[float, str, float]:
    closes = df["Close"].dropna()
    if len(closes) < 5:
        raise ValueError("Not enough bars to predict.")
    recent = closes.tail(lookback)
    x = np.arange(len(recent), dtype=float)
    y = recent.values
    m, b = np.polyfit(x, y, 1)
    trend_pred = float(m * len(x) + b)

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

    ema5 = recent.ewm(span=5, adjust=False).mean().iloc[-1]
    ema10 = recent.ewm(span=10, adjust=False).mean().iloc[-1]
    momentum_bias = (ema5 - ema10) * 0.06

    predicted = float(0.55 * trend_pred + 0.35 * ar_pred + momentum_bias)
    last_price = float(y[-1])
    change_pct = (predicted - last_price) / last_price * 100 if last_price else 0.0
    direction = "Up" if predicted > last_price else "Down" if predicted < last_price else "Flat"
    return predicted, direction, change_pct


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next close price for forex/gold using recent price history.")
    parser.add_argument("--symbol", default="XAUUSD", choices=list(DATA_SYMBOL_MAP.keys()), help="Market symbol to predict")
    parser.add_argument("--interval", default="5m", choices=["5m", "15m", "1h", "1d"], help="Data interval")
    parser.add_argument("--period", default="5m", help="History period for prediction")
    parser.add_argument("--lookback", type=int, default=12, help="Number of recent bars to use for prediction")
    args = parser.parse_args()

    df = load_data(args.symbol, args.period, args.interval)
    predicted, direction, change_pct = predict_next_price(df, lookback=args.lookback)
    last_close = float(df["Close"].dropna().iloc[-1])
    print(f"Symbol: {args.symbol}")
    print(f"Interval: {args.interval}")
    print(f"Last Close: {last_close:.5f}")
    print(f"Predicted next close: {predicted:.5f}")
    print(f"Direction: {direction}")
    print(f"Change: {change_pct:+.2f}%")


if __name__ == "__main__":
    main()

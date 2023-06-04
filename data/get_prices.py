import os
import yfinance as yf
import pandas as pd


def get_prices(tickers):
    start_date = pd.Timestamp('2010-01-01')
    end_date = pd.Timestamp.now() - pd.to_timedelta(1, unit='d')

    for ticker in tickers:
        if not os.path.exists(f"prices/{ticker}.pkl"):
            bars = yf.download(tickers=ticker,
                               interval="1d",
                               start=start_date,
                               end=end_date)
            print(bars)
            bars.to_csv(f"prices/{ticker}.csv")


if __name__ == "__main__":
    """
    Script to get stock prices from yfinance.
    -Prices are adjusted (using unscaled prices for model input is not recommended)
    -Prices only start from 2010-01-01 because that's when bulk twitter and google news starts
    """
    from utils import read_tickers

    tickers = read_tickers("sp500.csv")
    get_prices(tickers)

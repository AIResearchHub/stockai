import pandas as pd
import os


def to_csv(tickers):

    for ticker in tickers:

        try:
            bars = pd.read_pickle(f"prices/{ticker}.pkl")
            bars.to_csv(f"prices/{ticker}.csv")
            print(f"prices/{ticker}.csv")

            os.remove(f"prices/{ticker}.pkl")

        except Exception as e:
            print(e)


if __name__ == "__main__":
    from utils import read_tickers

    tickers = read_tickers("sp500.csv")
    to_csv(tickers)

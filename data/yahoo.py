import datetime as dt

import pandas as pd
from yahoo_finance import Share


def get_stock_prices(ticker, start_date, end_date=None):
    """Gets stock data from Yahoo Finance between dates, inclusive

    Args:
        ticker (str): The company ticker (e.g. GOOG)
        start_date (datetime.date): date to begin collecting share prices
        end_date (datetime.date): date to end collecting share prices. Default
            to today.
    Returns:
        A pd.DataFrame with the Date as the index and with columns
        ["Open", "Close", "High", "Low", "Volume", "Adj_Close"]
    """
    if end_date is None:
        end_date = dt.date.today()

    shares = Share(ticker)
    df = pd.DataFrame(shares.get_historical(start_date.isoformat(),
                                            end_date.isoformat()))
    return df.set_index("Date", drop=True) \
             .drop("Symbol", axis=1) \
             .astype(float) \
             .sort_index()

import math
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Input, Dense, Dropout

import yfinance as yf
"""**************  Technical Indicators **************"""

def calculate_boillinger_bands(data, window=10, num_of_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)

    return upper_band, lower_band


def calculate_rsi(data, window=10):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss

    rsi = 100 - (100/ (1 + rs))

    return rsi

def calculate_roc(data, periods=10):
    roc = ((data - data.shift(periods)) / data.shift(periods))
    return roc


"""***************************************************"""


# list of the tickers for the big tech companies

tickers = ["META", "AAPL", "MSFT", "AMZN", "GOOG"]

ticker_data_frames = []

stats = {}  # Initialize as a dictionary

for ticker in tickers:

    # Download the data using yfinance
    data = yf.download(ticker, period="1mo", interval="5m")  # Change period to '1mo'

    close = data['Close']

    upper, lower = calculate_boillinger_bands(close, window=14, num_of_std=2)
    width = upper - lower

    rsi = calculate_rsi(close, window=14)
    roc = calculate_roc(close, periods=14)

    volume = data["Volume"]
    diff = data["Close"].diff(1)
    percent_change_close = data['Close'].pct_change() * 100

    # create the dataframe for the current ticker and append it to the list
    ticker_df = pd.DataFrame({
        ticker + "_close": close,
        ticker + "_width": width,
        ticker + "_rsi": rsi,
        ticker + "_roc": roc,
        ticker + "_volume": volume,
        ticker + "_diff": diff,
        ticker + "_percent_change_close": percent_change_close

    })

    MEAN = ticker_df.mean()
    STD = ticker_df.std()

    # keep track of mean and std
    for column in MEAN.index:
        stats[f"{column}_mean"] = MEAN[column]
        stats[f"{column}_std"] = STD[column]
    
    # normalize the training features
    ticker_df = (ticker_df - MEAN) / STD

    ticker_data_frames.append(ticker_df)


# We have stats is in list form, lets convert it to the dataframe.

# stats_df = pd.DataFrame([stats], index=[0])

# print(stats_df.head())


df = pd.concat(ticker_data_frames, axis=1)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.dropna(inplace=True)

print(df.head())



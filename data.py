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


# In order to add the labels in our dataset we're going to lift out dataframe up by one position.
labels = df.shift(-1)

df = df.iloc[:-1]

labels = labels.iloc[:-1]

SEQUENCE_LEN  = 24 

def create_sequences(data, labels, mean, std, sequence_length=SEQUENCE_LEN):
    sequences = []
    lab = []
    data_size = len(data)


    for i in range(data_size - (sequence_length + 13)): # why 13 -> ensures that we have a data for the label.
        if i == 0:
            continue
        sequences.append(data[i:i+sequence_length])
        lab.append([labels[i-1], labels[i+12], mean, std])

    for i in range(0, len(lab)):
        last_price_data = sequences[i][-1][0]
        last_price_label = lab[i][0]

        if not last_price_data == last_price_label:
            print(f"Error: last_price_label = {last_price_label} and last_price_data = {last_price_data} are not equal")
    return np.array(sequences), np.array(lab)


sequence_dict = {}
sequnce_labels = {}

for ticker in tickers:

    # Extract close and volume data for every ticker

    close = df[ticker+'_close'].values
    volume = df[ticker+'_volume'].values
    rsi = df[ticker+'_rsi'].values
    roc = df[ticker+'_roc'].values
    width = df[ticker+'_width'].values
    diff = df[ticker+'_diff'].values
    pct_change = df[ticker+'_percent_change_close'].values

    ticker_data = np.column_stack((close,
                                   width,
                                   rsi,
                                   roc,
                                   volume,
                                   diff,
                                   pct_change))
    
    # Generate the sequence

    attribute = ticker+"_close"
    ticker_sequences, lab = create_sequences(
        ticker_data,
        labels[attribute].values[SEQUENCE_LEN-1:],
        stats[attribute+"_mean"],
        stats[attribute+"_std"]
    )

    sequence_dict[ticker] = ticker_sequences
    sequnce_labels[ticker] = lab



# lets aggregates the data and make  unified dataset for model training.

all_sequences = []
all_labels = []

for ticker in tickers:
    all_sequences.extend(sequence_dict[ticker])
    all_labels.extend(sequnce_labels[ticker])

# convert to numpy array
all_sequences = np.array(all_sequences)
all_labels = np.array(all_labels)

np.save("all_sequences.npy", all_sequences)
np.save("all_labels.npy", all_labels)


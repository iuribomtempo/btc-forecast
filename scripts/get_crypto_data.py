# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 08:54:34 2024

Scripto to fetch Binance OHLCV data 

@author: Iuri Bomtempo

Fetch 

"""

#%%
#Install Packages
!pip install python-binance
!pip install pandas
!pip install mplfinance
!pip install python-dotenv
!pip install dotenv

#%% Import packages
from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

#%% Load Binance API_KEY e SECRET_KEY from .env file

def get_binance_client():
    """
    Initialize and return an authenticated Binance Client instance.

    This function loads environment variables from a `.env` file using the `python-dotenv` package,
    and retrieves the `BINANCE_API_KEY` and `BINANCE_API_SECRET` to authenticate with the Binance API.

    Returns:
        binance.client.Client: Authenticated Binance API client.
    """
    load_dotenv()
    return Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_API_SECRET"))

client = get_binance_client()

#%% Def to retieve OHLCV data from binance
def fetch_binance_ohlcv(symbol, interval, start_str, end_str, save_path):
    client = get_binance_client()
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)

    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Salvo: {save_path}")

def fetch_batch_data(symbols, intervals, start_str="1 Jan 2017", end_str=None):
    if end_str is None:
        end_str = (datetime.now() - timedelta(days=1)).strftime("%d %b %Y")

    for symbol in symbols:
        for interval in intervals:
            filename = f"{symbol}_{interval}.csv"
            save_path = os.path.join("data", "raw", filename)
            fetch_binance_ohlcv(symbol, interval, start_str, end_str, save_path)

#%% Ex
if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT"]
    intervals = [
#        Client.KLINE_INTERVAL_1HOUR,
#        Client.KLINE_INTERVAL_4HOUR,
#        Client.KLINE_INTERVAL_1DAY,
        Client.KLINE_INTERVAL_1WEEK
        ]
    fetch_batch_data(symbols, intervals)


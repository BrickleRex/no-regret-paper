import yfinance as yf
import pandas as pd
import os
import json
from concurrent.futures import ThreadPoolExecutor
import time

# Define the ticker and date range
ticker = "AAPL"
start_date = "2000-01-01"
end_date = "2024-05-31"

def get_ticker_data(ticker=ticker, start_date=start_date, end_date=end_date, folder="OHLC"):
    # Download the data
    print(f"Downloading data for {ticker} from {start_date} to {end_date}")
    final_filename = f"{folder}/{ticker}_closing_prices.csv"

    if os.path.exists(final_filename):
        print(f"Data for {ticker} already exists. Skipping...")
        return
    
    data = yf.download(ticker, start=start_date, end=end_date)

    # Select only the closing prices
    closing_prices = data[['Close']]

    # Save to CSV
    os.makedirs(folder, exist_ok=True)
    closing_prices.to_csv(final_filename)

    print(f"Data downloaded and saved as {final_filename}")

with open("sp500_tickers.txt", "r") as f:
    tickers = f.read().splitlines()

for ticker in tickers:
    get_ticker_data(ticker)
    
print("All data downloaded successfully!")\


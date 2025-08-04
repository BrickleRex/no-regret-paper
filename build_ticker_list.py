import requests
import json
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import time
import random
from openai import OpenAI
import os
from dotenv import load_dotenv
from prompts import SECTOR_PROMPT

def get_static_tickers():
    with open('static_assets.txt', 'r') as f:
        static_tickers = f.read().splitlines()
    return static_tickers

def get_ticker_sector(ticker):
    # Load environment variables from .env file
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_KEY")
    if not openai_api_key:
        print("OPENAI_API_KEY not found in .env file.")
        return None

    client = OpenAI(api_key=openai_api_key)

    try:
        response = client.responses.create(
            model="gpt-4o",
            input=SECTOR_PROMPT.format(ticker=ticker),
            temperature=1e-4
        )
        # Extract the sector from the response
        sector = response.output_text
        # print(f"Sector for {ticker}: {sector}")
        return sector
    except Exception as e:
        print(f"Error fetching sector for {ticker}: {e}")
        return None

def get_sector_for_tickers(tickers):
    with ThreadPoolExecutor(max_workers=100) as executor:
        sectors = list(executor.map(get_ticker_sector, tickers))
    return {tickers[i]: sectors[i] for i in range(len(tickers))}

changes_csv = pd.read_csv('sp500_changes.csv')

all_changes = {}
#union of all processes rows (just split each row by ,)
for index, row in changes_csv.iterrows():
    changes = row['tickers']
    changes = changes.split(',')
    all_changes[row['date']] = changes
    
set_of_all_changes = set()
for changes in all_changes.values():
    set_of_all_changes.update(changes)
    
all_dynamic_tickers = list(set_of_all_changes)

print(all_dynamic_tickers)
print(f"length dynamic: {len(all_dynamic_tickers)}")

static_tickers = get_static_tickers()
print(f"length static: {len(static_tickers)}")

first_addition_dict = {}
last_seen_dict = {}

# Get the most recent date in the CSV to determine if ticker is still active
most_recent_date = changes_csv['date'].apply(lambda x: pd.to_datetime(x, format='%d/%m/%Y')).max()

for ticker in all_dynamic_tickers:
    # Find first addition
    for index, row in changes_csv.iterrows():
        if ticker in all_changes[row['date']]:
            first_addition_dict[ticker] = pd.to_datetime(row['date'], format='%d/%m/%Y').strftime('%d-%m-%Y')
            break
    
    # Find last seen (go through rows in reverse to find last occurrence)
    last_seen_date = None
    for index in reversed(changes_csv.index):
        row = changes_csv.iloc[index]
        if ticker in all_changes[row['date']]:
            last_seen_date = pd.to_datetime(row['date'], format='%d/%m/%Y')
            break
    
    # If last seen is the most recent date, ticker is still active
    if last_seen_date and last_seen_date == most_recent_date:
        last_seen_dict[ticker] = "N/A"  # Still active
    elif last_seen_date:
        last_seen_dict[ticker] = last_seen_date.strftime('%d-%m-%Y')  # Delisted - show last seen date
    else:
        last_seen_dict[ticker] = "N/A"  # Fallback

#save first_addition_dict to a csv
first_addition_df = pd.DataFrame(list(first_addition_dict.items()), columns=['ticker', 'date'])
first_addition_df.to_csv('first_additions.csv', index=False)

tickers_dict = []

dynamic_ticker_sectors = get_sector_for_tickers(all_dynamic_tickers)
static_ticker_sectors = get_sector_for_tickers(static_tickers)

for ticker in all_dynamic_tickers:
    if dynamic_ticker_sectors[ticker] is None:
        continue
    ticker_dict = {}
    ticker_dict['ticker'] = ticker
    ticker_dict['first_addition'] = first_addition_dict[ticker]
    ticker_dict['last_seen'] = last_seen_dict[ticker]
    ticker_dict["etf"] = False
    ticker_dict['sector'] = dynamic_ticker_sectors[ticker]
    ticker_dict["type"] = "dynamic"

    tickers_dict.append(ticker_dict)
    
for ticker in static_tickers:
    ticker_dict = {}
    ticker_dict['ticker'] = ticker
    ticker_dict['first_addition'] = "N/A"
    ticker_dict['last_seen'] = "N/A"
    ticker_dict["etf"] = True
    ticker_dict['sector'] = static_ticker_sectors[ticker]
    ticker_dict["type"] = "static"
    tickers_dict.append(ticker_dict)
    
#covnert to df then json
tickers_df = pd.DataFrame(tickers_dict)
tickers_df.to_json('tickers.json', orient='records')
    
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

def get_etf_tickers():
    with open('etf_list.json', 'r') as f:
        etf_tickers = json.load(f)
    return etf_tickers

etf_tickers = get_etf_tickers()
print(f"length etf: {len(etf_tickers)}")

tickers_dict = []

for ticker in etf_tickers:
    ticker_dict = {}
    ticker_dict['ticker'] = ticker['ticker']
    ticker_dict['first_addition'] = "N/A"
    ticker_dict['last_seen'] = "N/A"
    ticker_dict["etf"] = True
    ticker_dict['sector'] = ticker['sector']
    ticker_dict["type"] = "etf"
    ticker_dict["provider"] = ticker['provider']
    tickers_dict.append(ticker_dict)
    
#covnert to df then json
tickers_df = pd.DataFrame(tickers_dict)
tickers_df.to_json('tickers.json', orient='records', indent=4)
    
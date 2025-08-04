import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from metrics import compute_metrics, max_drawdown
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from prompts import get_fng_prompt
import traceback

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    df.dropna(inplace=True)

    print(f"Data ranges from {df.index[0]} to {df.index[-1]}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index is not a DatetimeIndex. Ensure that dates are parsed correctly.")

    return df

def bond_price_from_yield(yield_annual, coupon_annual, years=10):
    '''
    yield annual in decimal form (e.g. 0.05 for 5%)
    coupon annual in decimal absolute dollar form (e.g. 5 for $5)
    '''
    monthly_yield = yield_annual / 12.0
    months = years * 12
    monthly_coupon = coupon_annual / 12.0

    if monthly_yield == 0:
        pv_coupons = monthly_coupon * months
    else:
        pv_coupons = monthly_coupon * (1 - (1+monthly_yield)**(-months)) / monthly_yield

    pv_principal = 100 / ((1+monthly_yield)**months)
    price = pv_coupons + pv_principal
    return price

def get_rebalance_dates(df, frequency):
    if frequency == 'M':
        intended_dates = df.resample('M').last().index
    elif frequency == 'Q':
        intended_dates = df.resample('QE').last().index
    elif frequency == 'Y':
        # Use 'Y' year-end or 'YE' for year-end specifically
        # Here we use 'YE' to indicate year-end frequency.
        intended_dates = df.resample('YE').last().index
    elif frequency == 'S':
        intended_dates = df.resample('2QE').last().index
    elif frequency == '2Y':
        intended_dates = df.resample('2YE').last().index
    elif frequency == '3Y':
        intended_dates = df.resample('3YE').last().index
    elif frequency == 'W':
        intended_dates = df.resample('W').last().index
    elif frequency == 'D':
        intended_dates = df.resample('D').last().index
    else:
        raise ValueError("Invalid rebalance frequency. Use 'M', 'Q', or 'Y'.")

    actual_rebalance_dates = []
    for date in intended_dates:
        if date in df.index:
            actual_rebalance_dates.append(date)
        else:
            future_dates = df.index[df.index > date]
            if len(future_dates) > 0:
                actual_rebalance_dates.append(future_dates[0])

    return pd.Index(actual_rebalance_dates).unique()

def get_bond_returns(bond_yields, initial_coupon, years=10):
    bond_prices = [bond_price_from_yield(y/100.0, initial_coupon, years=years) for y in bond_yields]

    bond_returns = []
    for t in range(len(bond_prices)-1):
        coupon = initial_coupon / 252.0  # Daily coupon accrual for daily data
        r = (bond_prices[t+1] - bond_prices[t] + coupon) / bond_prices[t]
        bond_returns.append(r)

    return bond_returns

def correlation_coeff(nominal_arr1, nominal_arr2):
    returns1 = nominal_arr1.pct_change().dropna()
    returns2 = nominal_arr2.pct_change().dropna()
    return returns1.corr(returns2)
    

def plot_performance(results_dict, logger=None, rebalance_dates=None):
    fig, ax = plt.subplots(figsize=(10,6))
    for label, results in results_dict.items():
        ax.plot(results.index, results['Total_Nominal'], label=f'{label} (Nominal)')

    # if rebalance_dates is not None:
    #     ax.vlines(rebalance_dates, ymin=0, ymax=ax.get_ylim()[1], colors='red', linestyles='dashed', label='Rebalance Dates')

    ax.set_title('Portfolio Performance Comparison', fontsize=18)
    ax.set_xlabel('Date', fontsize=18)
    ax.set_ylabel('Value (higher is better)', fontsize=18)
    ax.legend(fontsize=12)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    logger.save_plot(fig, 'portfolio_performance_comparison.png')
    plt.tight_layout()
    plt.show()

def plot_metrics(results_dict, rfr=None, logger=None):
    volatilities = []
    max_drawdowns = []

    lbls = list(results_dict.keys())
    for lbl, res in results_dict.items():
        ann_ret, vol, mdd, metric, downside_dev, [sr, sortino, calmar, sorcal] = compute_metrics(res, rfr=rfr)
        print(f"{lbl}: Annualized Return={ann_ret:.2f}%, Volatility={vol:.4f}, Downside Deviation={downside_dev:.4f}, Max Drawdown={mdd:.4f}")
        print(f"Sharpe: {sr:.4f}, Sortino: {sortino:.4f}, Calmar: {calmar:.4f}, Sorcal: {sorcal:.4f}\n")
        volatilities.append(vol)
        max_drawdowns.append(mdd)

    x = np.arange(len(lbls))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.bar(x - width/2, volatilities, width=width, label='Volatility', color='blue')
    ax1.set_ylabel('Annualized Volatility (std dev, lower is better)', color='blue', fontsize=14)
    ax1.set_title('Volatility and Max Drawdown', fontsize=18)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lbls, fontsize=16)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=16)

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, [abs(mdd) for mdd in max_drawdowns], width=width, label='Max Drawdown', color='red')
    ax2.set_ylabel('Max Drawdown (absolute, lower is better)', color='red', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=16)

    lines, labels_ax1 = ax1.get_legend_handles_labels()
    lines2, labels_ax2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels_ax1 + labels_ax2, loc='upper left', fontsize=15)

    logger.save_plot(fig, 'volatility_max_drawdown_comparison.png')

    plt.tight_layout()
    plt.show()

def save_results_json(vars_dict, results_dict, logger):
    results = []
    
    rfr, vol, mdd, metric, downside_dev, [sr, sortino, calmar, sorcal] = compute_metrics(results_dict['Bonds Buy & Hold'])
    
    del results_dict['Bonds Buy & Hold']
    
    results.append({"Asset": "Bonds", "Return": rfr, "Volatility (lower is better)": vol, "Max Drawdown (lower is better)": mdd, "Sharpe Ratio (higher is better)": sr, "Sortino Ratio (higher is better)": sortino, "Calmar Ratio (higher is better)": calmar, "Sorcal Ratio (higher is better)": sorcal})
    
    for lbl, res in results_dict.items():
        ann_ret, vol, mdd, metric, downside_dev, [sr, sortino, calmar, sorcal] = compute_metrics(res, rfr=rfr) #compute metrics for each asset
        results.append({"Asset": lbl, "Return": ann_ret, "Volatility (lower is better)": vol, "Max Drawdown (lower is better)": mdd, "Downside Deviation (lower is better)": downside_dev, "Sharpe Ratio (higher is better)": sr, "Sortino Ratio (higher is better)": sortino, "Calmar Ratio (higher is better)": calmar, "Sorcal Ratio (higher is better)": sorcal})

    log_path = logger.get_log_dir()

    pd.DataFrame(results).to_json(f'{log_path}/results.json', index=False)

    #write vars_dict to json
    with open(f'{log_path}/vars.json', 'w') as f:
        json.dump(vars_dict, f)

    logger.write("Results saved to CSV files.")

def derisk_allocation(allocation: dict, percentage: float) -> dict:
    """
    Adjusts the allocation to move a specified percentage of non-Bonds/Cash assets into Bonds.

    Args:
        allocation (dict): The current portfolio allocation (asset: weight).
        percentage (float): The percentage (0.0 to 1.0) of assets to shift to Bonds.

    Returns:
        dict: The new, derisked allocation.
    """
    if not 0.0 <= percentage <= 1.0:
        raise ValueError("Percentage must be between 0.0 and 1.0.")

    new_allocation = allocation.copy()
    amount_to_shift_to_bonds = 0.0

    for asset, weight in list(new_allocation.items()): # Iterate over a list of items for safe modification
        if asset not in ['Bonds', 'Cash']:
            reduction = weight * percentage
            new_allocation[asset] = weight - reduction
            amount_to_shift_to_bonds += reduction
    
    if 'Bonds' not in new_allocation:
        new_allocation['Bonds'] = 0.0
        
    new_allocation['Bonds'] += amount_to_shift_to_bonds
    
    # Sanity check and re-normalize if necessary due to potential floating point issues
    current_sum = sum(new_allocation.values())
    if not np.isclose(current_sum, 1.0) and current_sum != 0: # Avoid division by zero if all weights became zero
        for asset_key in new_allocation:
            new_allocation[asset_key] /= current_sum
            
    return new_allocation

def get_aggregate_return_for_allocation_at_date(returns, allocation, date):
    #multiply weight from allocation with returns for each asset and sum for the date
    #make a copy of returns and add a col for cash returns, which is 0
    returns_copy = returns.copy()
    returns_copy['Cash'] = 0
    return sum(allocation[asset] * returns_copy.loc[date, asset] for asset in allocation if asset in returns_copy.columns)

def get_num_days_between_dates(df, date1, date2):
    #index first date and second date and return the number of rows between them, second date may not be in df so get the most recent date before it
    if date2 not in df.index:
        date2 = df.index[df.index < date2].max()
    return df.index.get_loc(date2) - df.index.get_loc(date1)

def should_derisk(df, current_date, current_allocation, cur_capital, threshold=0.05, lookahead_days=7):
    st = time.time()
    start_idx = df.index.get_loc(current_date)
    end_idx = start_idx + lookahead_days

    if end_idx >= len(df):
        end_idx = len(df) - 1
    
    window = df.iloc[start_idx:end_idx].pct_change().iloc[1:]
    
    #turn returns into nominal values
    # print(f"Current Date: {current_date}")
    # print(f"Starting Date: {df.index[start_idx]}")
    # print(f"Ending Date: {df.index[end_idx]}")
    # print(f"Window: {window}")
    # exit()
    nominal_values = [cur_capital]
    for date in window.index:
        nominal_values.append(nominal_values[-1] * (1 + get_aggregate_return_for_allocation_at_date(window, current_allocation, date)))
    
    returns = np.diff(nominal_values) / nominal_values[:-1]

    mdd = max_drawdown(nominal_values)
    downside_deviation = np.std(returns[returns < 0])
    en = time.time()
    # print(f"Time taken to check derisk: {en-st} seconds")
    # print(f"MDD: {mdd}, Threshold: {threshold}")
    return downside_deviation >= threshold, downside_deviation
    
def get_assets_by_cluster_string(cluster_string, tickers_file='tickers.json'):
    #cluster string is a string of alphabetic characters
    #using this list, we will read sectors.json to load all the corresponding sectors
    #then from tickers.json we will load all assets that have a sector in that set (each object has a sector and ticker key)
    #return a list of assets
    with open('sectors.json', 'r') as f:
        sectors = json.load(f)
    with open(tickers_file, 'r') as f:
        tickers = json.load(f)
    assets = []
    
    if cluster_string == "Z":
        assets = [ticker['ticker'] for ticker in tickers]# + ['Bonds', 'Cash', 'Gold']
    else:
        for sector in cluster_string:
            assets.extend([ticker['ticker'] for ticker in tickers if ticker['sector'] == sectors[sector]])
            print(f"Sector: {sectors[sector]}")
    
    return assets

def get_fng_for_date(date, n=1, k=1):
    #read fng.csv and return the value for the date
    #k is the interval between dates to return, so a n =4 and k=5 means we return ana rray of 4 values, each separated by 5 days
    df = pd.read_csv('fng.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    date = pd.to_datetime(date, format='%Y-%m-%d')
    
    if date not in df.index:
        # Find the last date in the index that is less than the given date
        possible_dates = df.index[df.index <= date]
        if len(possible_dates) == 0:
            raise ValueError(f"No available FNG data before {date.strftime('%Y-%m-%d')}")
        date = possible_dates[-1]
    
    date_idx_end = df.index.get_loc(date)
    date_idx_start = date_idx_end - (n * k)
    
    results = []
    for i in range(date_idx_end, date_idx_start, -k):
        # iloc expects integer row and integer column index, not column label
        # 'value' is the column name, so get its integer index
        value_col_idx = df.columns.get_loc('value')
        results.insert(0, df.iloc[i, value_col_idx])
    return results

def call_openai_with_timeout_retry(client, prompt, model="gpt-4.1-mini", timeout=30, max_retries=3, retry_delay=1):
    """
    Wrapper function to call OpenAI API with timeout and retry logic.
    
    Args:
        client: OpenAI client instance
        prompt: The prompt to send to the API
        model: The model to use (default: "gpt-4.1-mini")
        timeout: Timeout in seconds (default: 30)
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 1)
    
    Returns:
        The API response
    
    Raises:
        Exception: If all retry attempts fail or timeout occurs
    """
    for attempt in range(max_retries + 1):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    client.responses.create,
                    model=model,
                    input=prompt
                )
                response = future.result(timeout=timeout)
                return response.output_text
                
        except concurrent.futures.TimeoutError:
            if attempt < max_retries:
                print(f"OpenAI API call timed out (attempt {attempt + 1}/{max_retries + 1}). Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise Exception(f"OpenAI API call timed out after {max_retries + 1} attempts")
                
        except Exception as e:
            print(f"Error: {e}")
            if attempt < max_retries:
                print(f"OpenAI API call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise Exception(f"OpenAI API call failed after {max_retries + 1} attempts: {str(e)}")

def get_cluster_by_fng_response(date, client=None):
    fng_vals = get_fng_for_date(date, n=12, k=22)
    
    if client is None:
        print("No client provided, loading from .env")
        load_dotenv()
        client = OpenAI()
    
    
    response = call_openai_with_timeout_retry(client, get_fng_prompt(fng_vals), model="gpt-4.1", timeout=40, max_retries=5)
    #     response = client.responses.create(
    #         model="gpt-4.1-mini",
    #         input=get_fng_prompt(fng_vals),
    #     )
    # except Exception as e:
    #     print(f"Error: {e}")
    #     print(traceback.format_exc())
    #     exit()
    
    print(f"FNG Input: {str([int(x) for x in fng_vals])}")
    print(f"FNG Response: {response}")
    
    selected_sectors = response.split(',')
    selected_sectors = [sector.strip() for sector in selected_sectors]
    return selected_sectors

def get_cluster_string_from_sectors(sectors):
    #read sectors.json and return the string of sectors
    with open('sectors.json', 'r') as f:
        all_sectors = json.load(f)
            
    sector_to_key = {v: k for k, v in all_sectors.items()}
    return ''.join([sector_to_key[sector] for sector in sectors])

def get_common_sectors(sectors):
    #take a list of sector alphabetic characters and return the sectors freuqncy descending order
    full_string = ''.join(sectors)
    freq_count = ''.join(sorted(set(full_string), key=lambda x: full_string.count(x), reverse=True))
    
    #return the first 5 characters
    return freq_count[:3]

def hedge_cluster_string(cluster_string):
    with open('hedge_dict.json', 'r') as f:
        hedge_dict = json.load(f)
    
    with open('sectors.json', 'r') as f:
        sectors = json.load(f)
        
    hedge_sectors = []
    for sector in cluster_string:
        hedge_sectors.extend(hedge_dict[sectors[sector]])
        
    print(f"Hedge sectors: {hedge_sectors}")
        
    current_sectors = [sectors[sector] for sector in cluster_string]
    print(f"Current sectors: {current_sectors}")
    
    all_sectors = current_sectors + hedge_sectors
    all_sectors = list(set(all_sectors))
    print(f"All sectors: {all_sectors}")
    
    final_cluster_string = get_cluster_string_from_sectors(all_sectors)
    
    return final_cluster_string
    
def process_cluster_string(cluster_string, date, client=None, hedge=True, parallel_calls=5):
    if cluster_string != "dynamic":
        return cluster_string
    
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_calls) as executor:
        for i in range(parallel_calls):
            futures.append(executor.submit(get_cluster_by_fng_response, date, client=client))
        cluster_resp = [get_cluster_string_from_sectors(future.result()) for future in concurrent.futures.as_completed(futures)]
        
    final_sectors = get_common_sectors(cluster_resp)
    
    print(f"Final sectors: {final_sectors}")
    
    if hedge:
        final_sectors = hedge_cluster_string(final_sectors)
        
    return final_sectors

# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
# st = time.time()
# print(process_cluster_string('dynamic', '2015-01-01', client=client, hedge=True))
# en = time.time()
# print(f"Time taken: {en-st} seconds")
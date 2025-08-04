import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from utils import bond_price_from_yield, get_rebalance_dates, get_bond_returns, get_num_days_between_dates, should_derisk, get_assets_by_cluster_string, process_cluster_string
from scenario_utils import pick_best_scenario
from logger import Logger
from openai import OpenAI
from dotenv import load_dotenv
import os
from utils import get_fng_for_date
import random

# Set deterministic seeds for reproducibility
np.random.seed(42)
random.seed(42)

def calculate_leftover_cash(prices, units):
    target_units = {asset: int(unit) for asset, unit in units.items()}
    leftover_cash = {asset: (unit - target_units[asset])*prices[asset] for asset, unit in units.items()}

    target_units['Cash'] += sum(leftover_cash.values())
    target_values = {asset: target_units[asset] * prices[asset] for asset in units.keys()}

    return leftover_cash, target_units, target_values

def replace_symbol(symbol):
    replacements = {
        "Cash": "CASHX",
        "Bonds": "IEF",
        "Gold": "GLD",
        "SPX": "SPY"
    }
    return replacements.get(symbol, symbol)

def calculate_fee(fee_type, fee_pct, fee_per_share, old_units, new_units, new_prices):
    #calculate change in number of shares
    if fee_type is None:
        return 0.0
    
    # Calculate absolute change in units for each asset
    change_in_units_per_asset = {}
    for asset in set(list(old_units.keys()) + list(new_units.keys())):
        old_unit = old_units.get(asset, 0.0)
        new_unit = new_units.get(asset, 0.0)
        change_in_units_per_asset[asset] = abs(new_unit - old_unit)
    
    total_change_in_units = sum(change_in_units_per_asset.values())
    
    if total_change_in_units == 0.0:
        return 0.0
    
    if fee_type == "pct":
        # Calculate total dollar value of trades (both buys and sells)
        total_trade_value = 0.0
        for asset, units_changed in change_in_units_per_asset.items():
            if units_changed > 0 and asset in new_prices:
                total_trade_value += units_changed * new_prices[asset]
        return total_trade_value * fee_pct
    elif fee_type == "per_share":
        return total_change_in_units * fee_per_share
    else:
        return 0.0

def run_backtest(df, tickers_file, initial_capital=10000, rebalance_frequency='M', bond_years=10,
                 epsilon=5, k=30, objective="return", min_alloc=25, max_alloc=65, mode="backward", cluster_string=None, 
                 derisk_mode=False, derisk_threshold=0.05, derisk_duration=3, derisk_lookahead=14,
                 debug=True, rfr=None, exponential=False, logger=None, write_alloc=False, fee_type=None, fee_pct=0.0, fee_per_share=0.0, regularize=False, reg_factor=0.5,
                 percentile_threshold=None, lookback_days=None, lower_percentile_threshold=None, upper_percentile_threshold=None, hedge=True,
                 fng_threshold_high=None, fng_threshold_low=None, change_in_fng_threshold=None, set_all_cash_range=False, set_all_cash_change=False, use_mad_optimization=False):

    logger.log_variable("initial_capital", initial_capital)
    logger.log_variable("rebalance_frequency", rebalance_frequency)
    logger.log_variable("bond_years", bond_years)
    logger.log_variable("epsilon", epsilon)
    logger.log_variable("k", k)
    logger.log_variable("objective", objective)
    logger.log_variable("min_alloc", min_alloc)
    logger.log_variable("max_alloc", max_alloc)
    logger.log_variable("cluster_string", cluster_string)
    logger.log_variable("derisk_mode", derisk_mode)
    logger.log_variable("derisk_threshold", derisk_threshold)
    logger.log_variable("derisk_duration", derisk_duration)
    logger.log_variable("derisk_lookahead", derisk_lookahead)
    logger.log_variable("debug", debug)
    logger.log_variable("rfr", rfr)
    logger.log_variable("exponential", exponential)
    logger.log_variable("fee_type", fee_type)
    logger.log_variable("fee_pct", fee_pct)
    logger.log_variable("fee_per_share", fee_per_share)
    logger.log_variable("regularize", regularize)
    logger.log_variable("reg_factor", reg_factor)
    logger.log_variable("percentile_threshold", percentile_threshold)
    logger.log_variable("lookback_days", lookback_days)
    logger.log_variable("lower_percentile_threshold", lower_percentile_threshold)
    logger.log_variable("upper_percentile_threshold", upper_percentile_threshold)
    logger.log_variable("hedge", hedge)
    logger.log_variable("fng_threshold_high", fng_threshold_high)
    logger.log_variable("fng_threshold_low", fng_threshold_low)
    logger.log_variable("change_in_fng_threshold", change_in_fng_threshold)
    logger.log_variable("set_all_cash_range", set_all_cash_range)
    logger.log_variable("set_all_cash_change", set_all_cash_change)
    logger.log_variable("use_mad_optimization", use_mad_optimization)
    total_fee = 0.0
    
    # Store original cluster string to check against on each rebalancing date
    original_cluster_string = cluster_string
    
    baseline_allocation = {col: 0.0 for col in df.columns if col != 'CPI'}
    baseline_allocation['Cash'] = 1.0

    rebalance_dates = get_rebalance_dates(df, rebalance_frequency)

    bond_yields = df['Bonds'].values
    initial_coupon = bond_yields[0]
    
    bond_returns = get_bond_returns(bond_yields, initial_coupon, years=bond_years)

    initial_yield = df['Bonds'].iloc[0] / 100.0
    coupon_annual = df['Bonds'].iloc[0]
    initial_bond_price = bond_price_from_yield(initial_yield, coupon_annual, years=bond_years)

    initial_prices = df.iloc[0]

    initial_prices['Bonds'] = initial_bond_price
    initial_prices['Cash'] = 1.0

    ret_full = df.pct_change().iloc[1:].fillna(0)
    ret_full['Bonds'] = bond_returns

    allocation = baseline_allocation.copy()
    total_val = initial_capital
    
    DERISKED_ALLOCATION = {asset: 0.0 for asset in allocation.keys()}
    DERISKED_ALLOCATION['Bonds'] = 1.0

    assets_value = {asset: total_val * weight for asset, weight in allocation.items()}
    units = {asset: value / initial_prices[asset] for asset, value in assets_value.items()}

    results = pd.DataFrame(index=df.index, columns=['Total_Nominal', 'Total_Real', 'Bond_Yield'])

    best_allocations = []

    allocs_out = []
    
    is_derisked = False
    pre_derisk_allocation = allocation.copy()
    derisk_end_date = None
    derisk_history = []
    
    # Track previous prices for delisting handling
    previous_prices = initial_prices.copy()
    
    # Track blacklisted (delisted) assets to prevent future purchases
    blacklisted_assets = set()
    
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

    for date, row in df.iterrows():
        current_yield = row['Bonds'] / 100.0
        prices = row.copy()
        prices["Bonds"] = bond_price_from_yield(current_yield, coupon_annual, years=bond_years)
            
        prices["Cash"] = 1.0
        
        # Handle delisted assets (price = -1)
        delisted_assets = []
        for asset in units.keys():
            if units[asset] > 0 and prices.get(asset, 0) == -1:
                delisted_assets.append(asset)
        
        if delisted_assets:
            total_liquidated_value = 0.0
            
            for delisted_asset in delisted_assets:
                # Use previous valid price to calculate liquidated value
                liquidated_value = units[delisted_asset] * previous_prices[delisted_asset]
                total_liquidated_value += liquidated_value
                
                logger.write(f"Asset {delisted_asset} delisted on {date}. Liquidating {units[delisted_asset]:.4f} units at ${previous_prices[delisted_asset]:.2f} = ${liquidated_value:.2f}")
                
                # Add to blacklist to prevent future purchases
                blacklisted_assets.add(delisted_asset)
                logger.write(f"Added {delisted_asset} to blacklist - will not be selected in future rebalancing")
                
                # Remove from derisk allocation if present
                if delisted_asset in DERISKED_ALLOCATION:
                    DERISKED_ALLOCATION[delisted_asset] = 0.0
                    # Renormalize derisked allocation
                    total_derisk = sum(DERISKED_ALLOCATION.values())
                    if total_derisk > 0:
                        DERISKED_ALLOCATION = {k: v/total_derisk for k, v in DERISKED_ALLOCATION.items()}
                    
                # Clean allocations of delisted asset
                if delisted_asset in allocation:
                    allocation[delisted_asset] = 0.0
                if delisted_asset in pre_derisk_allocation:
                    pre_derisk_allocation[delisted_asset] = 0.0
                
                # Set delisted asset units to 0
                units[delisted_asset] = 0
            
            # Find other held assets to redistribute to (excluding delisted ones and those with 0 units)
            other_held_assets = [asset for asset in units.keys() 
                               if units[asset] > 0 and prices.get(asset, 0) != -1 and asset not in delisted_assets]
            
            if other_held_assets:
                # Distribute liquidated value uniformly among other held assets
                value_per_asset = total_liquidated_value / len(other_held_assets)
                
                for asset in other_held_assets:
                    additional_units = value_per_asset / prices[asset]
                    units[asset] += additional_units
                    
                logger.write(f"Redistributed ${total_liquidated_value:.2f} uniformly among {len(other_held_assets)} assets: {other_held_assets}")
                logger.write(f"Each asset received ${value_per_asset:.2f} in additional value")
            else:
                # If no other held assets, convert to cash
                cash_units = total_liquidated_value / prices['Cash']
                units['Cash'] += cash_units
                logger.write(f"No other held assets found. Converted ${total_liquidated_value:.2f} to cash ({cash_units:.4f} units)")
        
        # Update previous prices for next iteration (only for non-delisted assets)
        for asset in prices.index:
            if prices[asset] != -1:
                previous_prices[asset] = prices[asset]
        
        # Calculate portfolio value based on current holdings (units), not outdated assets_value keys
        values = {}
        for asset in units.keys():
            if units[asset] > 0 and asset in prices and prices[asset] != -1:
                values[asset] = units[asset] * prices[asset]
        
        total_val = sum(values.values())

        if derisk_mode:
            derisk_bool, mdd = should_derisk(df, date, allocation, total_val, threshold=derisk_threshold, lookahead_days=derisk_lookahead)
            if not is_derisked and derisk_bool:
                derisk_end_date = date + pd.DateOffset(days=derisk_duration)
                derisk_history.append((date, allocation, total_val))
                is_derisked = True
                pre_derisk_allocation = allocation.copy()
                allocation = DERISKED_ALLOCATION.copy()
                
                target_values = {asset: total_val * weight for asset, weight in allocation.items()}
                target_units = {asset: value / prices[asset] for asset, value in target_values.items()}

                fee = calculate_fee(fee_type, fee_pct, fee_per_share, units, target_units, prices)
                total_fee += fee
                total_val -= fee

                units = target_units
                assets_value = target_values
                
                logger.write(f"Derisked on {date} with allocation: {allocation} \n and total_val={total_val}\n because of MDD={mdd}")
            
            elif is_derisked and date >= derisk_end_date:
                derisk_bool, mdd = should_derisk(df, date, allocation, total_val, threshold=derisk_threshold, lookahead_days=derisk_lookahead)
                if not derisk_bool:
                    allocation = pre_derisk_allocation.copy()
                    is_derisked = False
                    
                    target_values = {asset: total_val * weight for asset, weight in allocation.items()}
                    target_units = {asset: value / prices[asset] for asset, value in target_values.items()}

                    fee = calculate_fee(fee_type, fee_pct, fee_per_share, units, target_units, prices)
                    total_fee += fee
                    total_val -= fee

                    units = target_units
                    assets_value = target_values
                    
                    logger.write(f"Underisked on {date} with allocation: {allocation} \n and total_val={total_val}")
                else:
                    derisk_end_date = date + pd.DateOffset(days=derisk_duration)
                    is_derisked = True
                    logger.write(f"Re-derisked on {date} with allocation: {allocation} \n and total_val={total_val}")

        if fng_threshold_high is not None or fng_threshold_low is not None or change_in_fng_threshold is not None:
            fng_filtering = True
        else:
            fng_filtering = False
        
        if fng_filtering:
            fng_for_date = get_fng_for_date(date, n=1, k=1)

            if change_in_fng_threshold is not None:
                change_period = change_in_fng_threshold[2]
                prev_fng_for_date = get_fng_for_date(date,n=change_period,k=1)
            else:
                prev_fng_for_date = None
        
        if date in rebalance_dates:
            if is_derisked:
                is_derisked = False
                allocation = pre_derisk_allocation.copy()
                logger.write(f"Underrisked on {date} with allocation: {allocation} \n and total_val={total_val}")
            
            # --- FNG threshold logic refactored ---
            if fng_filtering:
                should_skip, allocation, units, assets_value = handle_fng_threshold(
                    fng_for_date, fng_threshold_high, fng_threshold_low, allocation, units, assets_value, total_val, prices, set_all_cash_range, set_all_cash_change, logger=logger,
                    prev_fng_for_date=prev_fng_for_date, change_in_fng_threshold=change_in_fng_threshold
                )
                if should_skip:
                    logger.write(f"Skipping rebalancing on {date} because of FNG filtering")
                    continue
                
            #print the number of active assets (assets whose prices are not -1)
            active_assets = [asset for asset in prices.keys() if prices[asset] != -1]
            logger.write(f"Active assets on {date}: {len(active_assets)}")
            logger.write(f"Active assets: {active_assets}")
            
            bond_yields = df['Bonds'].values

            try:
                date_before_k = df.index[df.index <= date][-k]
            except:
                date_before_k = df.index[0]

            coupon_annual = df.loc[date_before_k, 'Bonds']

            bond_returns = get_bond_returns(bond_yields, coupon_annual, years=bond_years)
            ret_full['Bonds'] = bond_returns
            
            if original_cluster_string != "Z":
                print(f"Processing cluster string: {original_cluster_string}")
                current_cluster_string = process_cluster_string(original_cluster_string, date, client=client, hedge=hedge) + 'Q'
            else:
                current_cluster_string = original_cluster_string
            
            cluster = get_assets_by_cluster_string(current_cluster_string, tickers_file=tickers_file)
            active_cluster = [asset for asset in cluster if asset in active_assets]
            
            logger.write(f"Active cluster on {date}: {len(active_cluster)} assets")

            best_allocation = pick_best_scenario(df, allocation, epsilon, date, k, objective, ret_full, mode=mode, rebalance_dates=rebalance_dates, rebalance_frequency=rebalance_frequency, 
                                                 rfr=rfr, min_alloc=min_alloc, max_alloc=max_alloc, exponential=exponential, regularize=regularize, reg_factor=reg_factor, 
                                                 cluster=active_cluster, blacklist=blacklisted_assets, percentile_threshold=percentile_threshold, lookback_days=lookback_days, 
                                                 lower_percentile_threshold=lower_percentile_threshold, upper_percentile_threshold=upper_percentile_threshold, use_mad_optimization=use_mad_optimization)

            best_allocation = {asset: round(weight, 4) for asset, weight in best_allocation.items()}
            sum_alloc = sum(best_allocation.values())
            best_allocation = {asset: weight / sum_alloc for asset, weight in best_allocation.items()}
            
            best_allocations.append((date, best_allocation))

            if debug:
                logger.write(f"Rebalancing on {date} with best allocation: {best_allocation} with sum={sum(best_allocation.values())}")
                # Debug: Show current portfolio value breakdown
                current_values = {asset: units.get(asset, 0) * prices.get(asset, 0) for asset in units.keys() if units.get(asset, 0) > 0}
                logger.write(f"Portfolio before rebalancing: Total=${total_val:.2f}, Holdings: {len(current_values)} assets")

            target_values = {asset: total_val * weight for asset, weight in best_allocation.items()}
            target_units = {asset: value / prices[asset] for asset, value in target_values.items()}

            # leftover_cash, target_units, target_values = calculate_leftover_cash(prices, target_units)

            fee = calculate_fee(fee_type, fee_pct, fee_per_share, units, target_units, prices)
            total_fee += fee
            total_val -= fee
            
            if debug and fee > 0:
                logger.write(f"Rebalancing fee: ${fee:.2f} (Total fees so far: ${total_fee:.2f})")

            units = target_units
            allocation = best_allocation
            
            #log all the non zero weights in the allocation
            non_zero_weights = {asset: allocation[asset] for asset in allocation.keys() if allocation[asset] > 0}
            logger.write(f"Non-zero weights in allocation: {non_zero_weights}")
            
            assets_value = target_values

            if write_alloc:
                new_alloc = {}
                new_alloc["Start Date"] = date.strftime("%d/%m/%Y")
                for asset in list(allocation.keys()):

                    sym = replace_symbol(asset)

                    new_alloc[sym] = f"{round(allocation[asset]*100,2)}%"

                allocs_out.append(new_alloc)
                

            if total_val < 0:
                logger.write(f"Negative total value on {date} with total_val={total_val}, target_values={target_values}, prices={prices}, units={units}")
                break

                
        results.loc[date] = [total_val, 0, current_yield]

    logger.write(f"Backtest Complete: Starting capital: {initial_capital}, Final capital: {total_val}, Total fee: {total_fee}")

    if write_alloc:
        tag = ""
        if derisk_mode:
            tag = "derisked_"
        elif mode == "forward":
            tag = "forward_"
            
        logger.write_allocations(pd.DataFrame(allocs_out), tag=tag)
        logger.write_results(results, tag=tag)

    return results, best_allocations, rebalance_dates

def handle_fng_threshold(fng_for_date, fng_threshold_high, fng_threshold_low, allocation, units, assets_value, total_val, prices, set_all_cash_range, set_all_cash_change, logger, prev_fng_for_date=None, change_in_fng_threshold=None):
    """
    Checks FNG thresholds and optionally sets portfolio to all cash if not trading.
    Also checks for absolute change in FNG if change_in_fng_threshold is set (tuple: (min, max)).
    Returns (should_skip, allocation, units, assets_value)
    """
    should_skip = False
    # FNG value threshold logic
    print(f"fng for date: {fng_for_date}, fng threshold high: {fng_threshold_high}, fng threshold low: {fng_threshold_low}")
    
    if fng_threshold_high is not None and fng_threshold_low is not None:
        if (fng_for_date[0] < fng_threshold_low or fng_for_date[0] > fng_threshold_high):
            logger.write(f"[FNG] FNG {fng_for_date[0]} is not extreme enough (not below {fng_threshold_low} or above {fng_threshold_high}), going to cash and skipping")
            should_skip = "range"
    elif fng_threshold_high is not None:
        if (fng_for_date[0] > fng_threshold_high):
            logger.write(f"[FNG] FNG {fng_for_date[0]} is not above high threshold {fng_threshold_high}, going to cash and skipping")
            should_skip = "range"
    elif fng_threshold_low is not None:
        if (fng_for_date[0] < fng_threshold_low):
            logger.write(f"[FNG] FNG {fng_for_date[0]} is not below low threshold {fng_threshold_low}, going to cash and skipping")
            should_skip = "range"
    else:
        logger.write("[FNG] FNG thresholds are not set, skipping FNG check")
    # FNG change threshold logic
    print(f"should skip: {should_skip}, change in fng threshold: {change_in_fng_threshold}, prev fng for date: {prev_fng_for_date}")
    
    if not should_skip and change_in_fng_threshold is not None and prev_fng_for_date is not None:
        min_change, max_change, _ = change_in_fng_threshold
        abs_change = abs(fng_for_date[0] - prev_fng_for_date[0])
        if not (min_change <= abs_change <= max_change):
            logger.write(f"[FNG] Absolute change in FNG ({abs_change}) is not within bounds ({min_change}, {max_change}), going to cash and skipping")
            should_skip = "change"
        else:
            logger.write(f"[FNG] Absolute change in FNG ({abs_change}) is within bounds ({min_change}, {max_change}), not going to cash")
    elif not should_skip:
        logger.write("[FNG] FNG change thresholds are not set, skipping FNG change check")
        
    if should_skip == "range" and set_all_cash_range:
        # Set allocation to all cash
        allocation = {asset: 0.0 for asset in allocation}
        allocation['Cash'] = 1.0
        # Set units to all cash
        total_cash = total_val
        units = {asset: 0.0 for asset in units}
        units['Cash'] = total_cash / prices['Cash']
        # Set assets_value to all cash
        assets_value = {asset: 0.0 for asset in assets_value}
        assets_value['Cash'] = total_cash
        logger.write(f"[FNG] Portfolio set to all cash.")
    elif should_skip == "change" and set_all_cash_change:
        # Set allocation to all cash
        allocation = {asset: 0.0 for asset in allocation}
        allocation['Cash'] = 1.0
        # Set units to all cash
        total_cash = total_val
        units = {asset: 0.0 for asset in units}
        units['Cash'] = total_cash / prices['Cash']
        # Set assets_value to all cash
        assets_value = {asset: 0.0 for asset in assets_value}
        assets_value['Cash'] = total_cash
        logger.write(f"[FNG] Portfolio set to all cash.")
    else:
        logger.write(f"FNG for date: {fng_for_date} is within thresholds")
    return should_skip, allocation, units, assets_value

def run_spx_buy_and_hold(df, initial_capital=10000, logger=None):
    initial_spx_price = df['SPX'].iloc[0]
    initial_cpi = df['CPI'].iloc[0]

    units = initial_capital / initial_spx_price
    results = pd.DataFrame(index=df.index, columns=['Total_Nominal','Total_Real', 'Bond_Yield'])

    for date, row in df.iterrows():
        current_yield = row['Bonds']/100.0
        spx_price = row['SPX']
        cpi = row['CPI']
        total_val = units * spx_price
        total_real = total_val / (cpi / initial_cpi)
        results.loc[date] = [total_val, total_real, current_yield]

    logger.write(f"Buy & Hold SPX: Starting capital: {initial_capital}, Final capital: {total_val}")

    return results

def run_gold_buy_and_hold(df, initial_capital=10000, logger=None):
    initial_gold_price = df['Gold'].iloc[0]
    initial_cpi = df['CPI'].iloc[0]

    units = initial_capital / initial_gold_price
    results = pd.DataFrame(index=df.index, columns=['Total_Nominal','Total_Real', 'Bond_Yield'])

    for date, row in df.iterrows():
        current_yield = row['Bonds']/100.0
        gold_price = row['Gold']
        cpi = row['CPI']
        total_val = units * gold_price
        total_real = total_val / (cpi / initial_cpi)
        results.loc[date] = [total_val, total_real, current_yield]
    
    logger.write(f"Buy & Hold Gold: Starting capital: {initial_capital}, Final capital: {total_val}")

    return results

def run_bonds_buy_and_hold(df, initial_capital=10000, years=10, logger=None):
    initial_yield = df['Bonds'].iloc[0]/100.0
    coupon_annual = df['Bonds'].iloc[0]
    current_capital = initial_capital
    
    # Calculate how many bonds we can buy (bonds have $100 par value)
    par_value = 1000
    num_bonds = initial_capital / par_value
    
    initial_cpi = df['CPI'].iloc[0]
    results = pd.DataFrame(index=df.index, columns=['Total_Nominal','Total_Real', 'Bond_Yield'])
    
    # Track accumulated coupons
    accumulated_coupons = 0.0
    
    # Calculate daily coupon accrual
    daily_coupon = ((coupon_annual / 100) / 252.0)*par_value*num_bonds
    print(f"Daily coupon: {daily_coupon}")
    # exit()
    
    # Calculate maturity date
    maturity_date = df.index[0] + pd.DateOffset(years=years)
    
    for date, row in df.iterrows():
        current_yield = row['Bonds']/100.0
        cpi = row['CPI']
        
        # Accumulate daily coupon
        accumulated_coupons += daily_coupon
        
        # Calculate total value
        if date >= maturity_date:
            # Bond has matured, get principal back
            total_val = num_bonds * par_value + accumulated_coupons
            num_bonds = total_val / par_value
            daily_coupon = ((coupon_annual / 100) / 252.0)*par_value*num_bonds
            accumulated_coupons = 0.0
            maturity_date = date + pd.DateOffset(years=years)
            current_capital = total_val
            logger.write(f"Bond matured on {date}: Reinvested ${total_val:.2f} into {num_bonds:.2f} bonds at ${par_value:.2f} each (yield: {current_yield*100:.2f}%)")
        else:
            # Before maturity, value is just accumulated coupons (no mark to market)
            # Principal is locked up until maturity
            total_val = current_capital + accumulated_coupons
        
        total_real = total_val / (cpi / initial_cpi)
        results.loc[date] = [total_val, total_real, current_yield]
        
    logger.write(f"Buy & Hold Bonds: Starting capital: {initial_capital}, Final capital: {total_val}")
    return results
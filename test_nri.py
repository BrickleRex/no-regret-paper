import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from backtest import run_backtest, run_bonds_buy_and_hold, run_spx_buy_and_hold, run_gold_buy_and_hold
from utils import load_data, plot_performance, plot_metrics, correlation_coeff, save_results_json, get_assets_by_cluster_string
from metrics import compute_metrics
from logger import Logger
import json
import traceback
import random
import sys
import os

# Set deterministic seeds for reproducibility
np.random.seed(42)
random.seed(42)

def grid_search_parameters(train_df, rebalance_frequency, objective,
                           k_start, k_end, k_step, min_alloc_start=25, max_alloc_start=30,
                           min_alloc_step=5, max_alloc_step=5,
                           min_alloc_end=15, max_alloc_end=85,
                           initial_capital=10000, years=10, logger=None):
    best_epsilon = None
    best_k = None
    best_score = None
    best_allocation = None
    best_min_alloc = None
    best_max_alloc = None

    epsilon = 5

    # for epsilon in range(epsilon_start, epsilon_end+1, epsilon_step):
    total_iterations = len(range(k_start, k_end+1, k_step)) * len(range(min_alloc_start, min_alloc_end+1, min_alloc_step)) * len(range(max_alloc_start, max_alloc_end+1, max_alloc_step))
    logger.write(f"Total iterations: {total_iterations}")

    for k in range(k_start, k_end+1, k_step):
        for min_alloc in range(min_alloc_start, min_alloc_end+1, min_alloc_step):
            for max_alloc in range(max_alloc_start, max_alloc_end+1, max_alloc_step):
                logger.write(f"Trying min_alloc={min_alloc}, max_alloc={max_alloc}")
                results, allocation = run_backtest(train_df, initial_capital=initial_capital, rebalance_frequency=rebalance_frequency,
                                                    years=years, epsilon=epsilon, k=k, objective=objective, min_alloc=min_alloc, max_alloc=max_alloc)
                
                ann_return, vol, mdd, metric, downside_dev, _ = compute_metrics(results, objective=objective)
                # Use annualized return as the optimization metric:
                score = metric
                if (best_score is None) or (score > best_score):
                    best_score = score
                    best_epsilon = epsilon
                    best_k = k
                    best_allocation = allocation
                    best_min_alloc = min_alloc
                    best_max_alloc = max_alloc

    logger.write(f"Best parameters on training set: epsilon={best_epsilon}, k={best_k} with annualized return={best_score:.2f}%, min_alloc={best_min_alloc}, max_alloc={best_max_alloc}")
    logger.write(f"Best allocation: {best_allocation}")
    return best_epsilon, best_k, best_min_alloc, best_max_alloc

def grid_search_parameters_only_derisk(train_df, rebalance_frequency, objective,
                           derisk_threshold_start, derisk_threshold_end, derisk_threshold_step,
                           derisk_duration_start, derisk_duration_end, derisk_duration_step,
                           derisk_lookahead_start, derisk_lookahead_end, derisk_lookahead_step,
                           epsilon=20, k=60, min_alloc=35, max_alloc=80,
                           initial_capital=10000, years=10, logger=None, rfr=2.67, exponential=False,
                           percentile_threshold=None, lookback_days=None):
    
    best_derisk_threshold = None
    best_derisk_duration = None
    best_derisk_lookahead = None
    best_score = None
    best_allocation = None
    
    grid_results = []
    
    #calculate number of iterations by zipping the ranges then filtering out the ones where derisk_duration <= derisk_lookahead
    iterations = list(zip(np.arange(derisk_threshold_start, derisk_threshold_end+1, derisk_threshold_step),
                          np.arange(derisk_duration_start, derisk_duration_end+1, derisk_duration_step),
                          np.arange(derisk_lookahead_start, derisk_lookahead_end+1, derisk_lookahead_step)))
    iterations = [i for i in iterations if i[1] > i[2]]
    
    total_iterations = len(iterations)
    logger.write(f"Total iterations: {total_iterations}\n\n==================\n")
    
    for derisk_threshold in np.arange(derisk_threshold_start, derisk_threshold_end+1, derisk_threshold_step):
        for derisk_duration in np.arange(derisk_duration_start, derisk_duration_end+1, derisk_duration_step):
            for derisk_lookahead in np.arange(derisk_lookahead_start, derisk_lookahead_end+1, derisk_lookahead_step):
                if derisk_duration <= derisk_lookahead:
                    logger.write(f"Trying derisk_threshold={derisk_threshold}, derisk_duration={derisk_duration}, derisk_lookahead={derisk_lookahead}")
                    results, allocation = run_backtest(train_df, initial_capital=10000, rebalance_frequency=rebalance_frequency, mode="backward", derisk_mode=True, 
                                        derisk_threshold=derisk_threshold, derisk_duration=int(derisk_duration), derisk_lookahead=int(derisk_lookahead),
                                        bond_years=10, epsilon=20, k=k, objective=objective, min_alloc=min_alloc, max_alloc=max_alloc, debug=True, rfr=rfr, exponential=exponential, logger=logger, write_alloc=True,
                                        percentile_threshold=percentile_threshold, lookback_days=lookback_days)
                    
                    ann_return, vol, mdd, metric, downside_dev, [sr, sortino, calmar, sorcal] = compute_metrics(results, objective=objective)
                    # Use annualized return as the optimization metric:
                    score = ann_return
                    print(f"Derisked portfolio performance with epsilon={20}, k={k}: Annualized Return={ann_return:.2f}%, Vol={vol:.4f}, MDD={mdd:.4f} (Sharpe={sr:.4f}, Sortino={sortino:.4f}, Calmar={calmar:.4f})")
                    
                    grid_results.append({
                        "derisk_threshold": derisk_threshold,
                        "derisk_duration": derisk_duration,
                        "derisk_lookahead": derisk_lookahead,
                        "ann_return": ann_return,
                        "vol": vol,
                        "mdd": mdd,
                    })
                    
                    if (best_score is None) or (score > best_score):
                        best_score = score
                        best_derisk_threshold = derisk_threshold
                        best_derisk_duration = derisk_duration
                        best_derisk_lookahead = derisk_lookahead
                        best_allocation = allocation

    logger.write(f"Best parameters on training set: derisk_threshold={best_derisk_threshold}, derisk_duration={best_derisk_duration}, derisk_lookahead={best_derisk_lookahead} with annualized return={best_score:.2f}%")
    logger.write(f"Best allocation: {best_allocation}")
    return grid_results

def run_complete_backtest(universe, start_date, end_date, tickers_file, rebalance_frequency, objective, k, min_alloc, max_alloc, exponential, cluster_string, test_derisk, derisk_threshold, derisk_duration, derisk_lookahead,
                          regularize, reg_factor, percentile_threshold=None, lower_percentile_threshold=None, upper_percentile_threshold=None, 
                          lookback_days=None, write_csv=False, hedge=True,
                          fng_threshold_high=None, fng_threshold_low=None, change_in_fng_threshold=None, set_all_cash_range=False, set_all_cash_change=False, use_mad_optimization=False, use_hedge_optimization=False):
    hedge_str = "hedged" if hedge else "unhedged"
    fng_str = ""
    if fng_threshold_high is not None:
        fng_str += f"fng_thresh_high_{fng_threshold_high}_"
    else:
        fng_str += "fng_thresh_high_None_"
    if fng_threshold_low is not None:
        fng_str += f"fng_thresh_low_{fng_threshold_low}_"
    else:
        fng_str += "fng_thresh_low_None_"
    if change_in_fng_threshold is not None:
        fng_str += f"fng_thresh_change_{change_in_fng_threshold}"
    else:
        fng_str += "fng_thresh_change_None"
        
    percentile_threshold_str = f"percentile_{percentile_threshold}_lookback_{lookback_days}" if percentile_threshold is not None else "percentile_None_lookback_None"
    
    opt_methods = []
    if use_mad_optimization:
        opt_methods.append("mad")
    if use_hedge_optimization:
        opt_methods.append("hedge")
    if not opt_methods:
        opt_methods.append("scenario")
    opt_str = "_".join(opt_methods)
        
    logger = Logger(tag=f"{rebalance_frequency}_{objective}_{k}_{cluster_string}_{hedge_str}_{fng_str}_{percentile_threshold_str}_{opt_str}", writer=True)
    
    universe_to_filename = {"SPDR500": "combined_data3.csv", "SECTOR-ETFs": "combined_stock_data_etfs.csv", "COCKROACH": "combined_data2.csv"}

    data_file = universe_to_filename[universe]  # Replace with your file
    df_raw = load_data(data_file)

    # assets = get_assets_by_cluster_string(cluster_string)
    # assets += ["Gold", "Bonds"]
    
    # print(f"Length of assets: {len(assets)}")
    # df_raw = df_raw[assets]

    print(f"Shape of raw data: {df_raw.shape}")
    # Training period: 2010 to 2020
    # train_df = df_raw.loc['2010-01-01':'2018-12-31']
    # Testing period: 2020 to 2024
    test_df = df_raw.loc[start_date:end_date]

    print(f"Shape of test data: {test_df.shape}")
    
    # derisk_threshold_start = 0.01
    # derisk_threshold_end = 0.51
    # derisk_threshold_step = 0.05
    # derisk_duration_start = 1.0
    # derisk_duration_end = 31.0
    # derisk_duration_step = 1.0
    # derisk_lookahead_start = 1.0
    # derisk_lookahead_end = 31.0
    # derisk_lookahead_step = 1.0
    # exit()

    logger.log_variable("universe", universe)
    logger.log_variable("start_date", start_date)
    logger.log_variable("end_date", end_date)
    logger.log_variable("tickers_file", tickers_file)
    logger.log_variable("rebalance_frequency", rebalance_frequency)
    logger.log_variable("objective", objective)
    logger.log_variable("k", k)
    logger.log_variable("min_alloc", min_alloc)
    logger.log_variable("max_alloc", max_alloc)
    logger.log_variable("exponential", exponential)
    logger.log_variable("cluster_string", cluster_string)
    logger.log_variable("derisk_threshold", derisk_threshold)
    logger.log_variable("derisk_duration", derisk_duration)
    logger.log_variable("derisk_lookahead", derisk_lookahead)
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
    logger.log_variable("use_hedge_optimization", use_hedge_optimization)
    logger.write(f"Shape of test data: {test_df.shape}")
    
    vars_dict = {
        "universe": universe,
        "start_date": start_date,
        "end_date": end_date,
        "tickers_file": tickers_file,
        "rebalance_frequency": rebalance_frequency,
        "objective": objective,
        "k": k,
        "min_alloc": min_alloc,
        "max_alloc": max_alloc,
        "exponential": exponential,
        "cluster_string": cluster_string,
        "derisk_threshold": derisk_threshold,
        "derisk_duration": derisk_duration,
        "derisk_lookahead": derisk_lookahead,
        "regularize": regularize,
        "reg_factor": reg_factor,
        "percentile_threshold": percentile_threshold,
        "lookback_days": lookback_days,
        "lower_percentile_threshold": lower_percentile_threshold,
        "upper_percentile_threshold": upper_percentile_threshold,
        "hedge": hedge,
        "fng_threshold_high": fng_threshold_high,
        "fng_threshold_low": fng_threshold_low,
        "change_in_fng_threshold": change_in_fng_threshold,
        "set_all_cash_range": set_all_cash_range,
        "set_all_cash_change": set_all_cash_change,
        "use_mad_optimization": use_mad_optimization,
        "use_hedge_optimization": use_hedge_optimization,
    }
    
    # logger.write(f"Variables: {vars_dict}")

    logger.write("================\nBonds Buy & Hold\n================")
    bonds_results = run_bonds_buy_and_hold(test_df, initial_capital=10000, years=10, logger=logger)

    rfr, _, _, _, _, _ = compute_metrics(bonds_results)

    logger.log_variable("rfr", rfr)

    logger.write("================\nSPX Buy & Hold\n================")
    spx_results = run_spx_buy_and_hold(test_df, initial_capital=10000, logger=logger)

    logger.write("================\nGold Buy & Hold\n================")
    gold_results = run_gold_buy_and_hold(test_df, initial_capital=10000, logger=logger)

    logger.write("================\nRegret Learning (Scenario-based)\n================")
    st = time.time()
    regret_results, _, regret_rebalance_dates = run_backtest(test_df, tickers_file=tickers_file, initial_capital=10000, rebalance_frequency=rebalance_frequency, cluster_string=cluster_string,
                                     bond_years=10, epsilon=20, k=k, objective=objective, min_alloc=min_alloc, max_alloc=max_alloc, debug=True, rfr=rfr, 
                                     exponential=exponential, logger=logger, write_alloc=True, fee_type="per_share", fee_pct=0.0, fee_per_share=0.00, regularize=regularize, reg_factor=reg_factor,
                                     percentile_threshold=percentile_threshold, lookback_days=lookback_days, lower_percentile_threshold=lower_percentile_threshold,
                                     upper_percentile_threshold=upper_percentile_threshold, hedge=hedge,
                                     fng_threshold_high=fng_threshold_high, fng_threshold_low=fng_threshold_low, change_in_fng_threshold=change_in_fng_threshold,
                                     set_all_cash_range=set_all_cash_range, set_all_cash_change=set_all_cash_change, use_mad_optimization=False, use_hedge_optimization=False)
    
    test_ann_return, test_vol, test_mdd, test_downside_deviation, metric, [sr, sortino, calmar, sorcal] = compute_metrics(regret_results, objective=objective, rfr=rfr)
    en = time.time()
    logger.write(f"Scenario-based performance with epsilon={20}, k={k}: Annualized Return={test_ann_return:.2f}%, Vol={test_vol:.4f}, Downside Deviation={test_downside_deviation:.4f}, MDD={test_mdd:.4f} (Sharpe={sr:.4f}, Sortino={sortino:.4f}, Calmar={calmar:.4f}, Sorcal={sorcal:.4f}) in {en-st:.2f} seconds")
    
    # MAD Optimization approach (if enabled)
    mad_results = None
    derisked_mad_results = None
    derisked_hedge_results = None
    if use_mad_optimization:
        logger.write("================\nMAD Optimization\n================")
        st = time.time()
        mad_results, _, _ = run_backtest(test_df, tickers_file=tickers_file, initial_capital=10000, rebalance_frequency=rebalance_frequency, cluster_string=cluster_string,
                                         bond_years=10, epsilon=20, k=k, objective=objective, min_alloc=min_alloc, max_alloc=max_alloc, debug=True, rfr=rfr, 
                                         exponential=exponential, logger=logger, write_alloc=True, fee_type="per_share", fee_pct=0.0, fee_per_share=0.00, regularize=regularize, reg_factor=reg_factor,
                                         percentile_threshold=percentile_threshold, lookback_days=lookback_days, lower_percentile_threshold=lower_percentile_threshold,
                                         upper_percentile_threshold=upper_percentile_threshold, hedge=hedge,
                                         fng_threshold_high=fng_threshold_high, fng_threshold_low=fng_threshold_low, change_in_fng_threshold=change_in_fng_threshold,
                                         set_all_cash_range=set_all_cash_range, set_all_cash_change=set_all_cash_change, use_mad_optimization=True, use_hedge_optimization=False)
        
        mad_ann_return, mad_vol, mad_mdd, mad_downside_deviation, mad_metric, [mad_sr, mad_sortino, mad_calmar, mad_sorcal] = compute_metrics(mad_results, objective=objective, rfr=rfr)
        en = time.time()
        logger.write(f"MAD optimization performance with k={k}: Annualized Return={mad_ann_return:.2f}%, Vol={mad_vol:.4f}, Downside Deviation={mad_downside_deviation:.4f}, MDD={mad_mdd:.4f} (Sharpe={mad_sr:.4f}, Sortino={mad_sortino:.4f}, Calmar={mad_calmar:.4f}, Sorcal={mad_sorcal:.4f}) in {en-st:.2f} seconds")
    
    # Hedge (Multiplicative Weights) Optimization approach (if enabled)
    hedge_results = None
    derisked_hedge_results = None
    if use_hedge_optimization:
        logger.write("================\nHedge (Multiplicative Weights) Optimization\n================")
        st = time.time()
        hedge_results, _, _ = run_backtest(test_df, tickers_file=tickers_file, initial_capital=10000, rebalance_frequency=rebalance_frequency, cluster_string=cluster_string,
                                           bond_years=10, epsilon=20, k=k, objective=objective, min_alloc=min_alloc, max_alloc=max_alloc, debug=True, rfr=rfr, 
                                           exponential=exponential, logger=logger, write_alloc=True, fee_type="per_share", fee_pct=0.0, fee_per_share=0.00, regularize=regularize, reg_factor=reg_factor,
                                           percentile_threshold=percentile_threshold, lookback_days=lookback_days, lower_percentile_threshold=lower_percentile_threshold,
                                           upper_percentile_threshold=upper_percentile_threshold, hedge=hedge,
                                           fng_threshold_high=fng_threshold_high, fng_threshold_low=fng_threshold_low, change_in_fng_threshold=change_in_fng_threshold,
                                           set_all_cash_range=set_all_cash_range, set_all_cash_change=set_all_cash_change, use_mad_optimization=False, use_hedge_optimization=True)
        
        hedge_ann_return, hedge_vol, hedge_mdd, hedge_downside_deviation, hedge_metric, [hedge_sr, hedge_sortino, hedge_calmar, hedge_sorcal] = compute_metrics(hedge_results, objective=objective, rfr=rfr)
        en = time.time()
        logger.write(f"Hedge optimization performance with k={k}: Annualized Return={hedge_ann_return:.2f}%, Vol={hedge_vol:.4f}, Downside Deviation={hedge_downside_deviation:.4f}, MDD={hedge_mdd:.4f} (Sharpe={hedge_sr:.4f}, Sortino={hedge_sortino:.4f}, Calmar={hedge_calmar:.4f}, Sorcal={hedge_sorcal:.4f}) in {en-st:.2f} seconds")
    
    # derisk_results = grid_search_parameters_only_derisk(test_df, rebalance_frequency, objective, derisk_threshold_start, derisk_threshold_end, derisk_threshold_step, derisk_duration_start, derisk_duration_end, derisk_duration_step, derisk_lookahead_start, derisk_lookahead_end, derisk_lookahead_step, epsilon=20, k=k, min_alloc=min_alloc, max_alloc=max_alloc, initial_capital=10000, years=10, logger=logger, rfr=rfr, exponential=exponential)
    # derisk_results_df = pd.DataFrame(derisk_results)
    # derisk_results_df.to_csv(f"derisk_results_{rebalance_frequency}_{objective}_{k}_{min_alloc}_{max_alloc}.csv", index=False)
    
    if test_derisk:
        logger.write("================\nDerisked Portfolio (Scenario-based)\n================")
        st = time.time()
        derisked_results, _ = run_backtest(test_df, tickers_file=tickers_file, initial_capital=10000, rebalance_frequency=rebalance_frequency, cluster_string=cluster_string, mode="backward", derisk_mode=True, 
                                        derisk_threshold=derisk_threshold, derisk_duration=derisk_duration, derisk_lookahead=derisk_lookahead,
                                        bond_years=10, epsilon=20, k=k, objective=objective, min_alloc=min_alloc, max_alloc=max_alloc, debug=True, rfr=rfr, exponential=exponential, logger=logger, write_alloc=True, fee_type="per_share", fee_pct=0.0, fee_per_share=0.0035,
                                        percentile_threshold=percentile_threshold, lookback_days=lookback_days, lower_percentile_threshold=lower_percentile_threshold,
                                        upper_percentile_threshold=upper_percentile_threshold, hedge=hedge,
                                        fng_threshold_high=fng_threshold_high, fng_threshold_low=fng_threshold_low, change_in_fng_threshold=change_in_fng_threshold,
                                        set_all_cash_range=set_all_cash_range, set_all_cash_change=set_all_cash_change, use_mad_optimization=False, use_hedge_optimization=False)
        
        derisked_ann_return, derisked_vol, derisked_mdd, derisked_downside_deviation, derisked_metric, [derisked_sr, derisked_sortino, derisked_calmar, derisked_sorcal] = compute_metrics(derisked_results, objective=objective, rfr=rfr)
        en = time.time()
        logger.write(f"Derisked scenario-based performance with epsilon={20}, k={k}: Annualized Return={derisked_ann_return:.2f}%, Vol={derisked_vol:.4f}, MDD={derisked_mdd:.4f} (Sharpe={derisked_sr:.4f}, Sortino={derisked_sortino:.4f}, Calmar={derisked_calmar:.4f}, Sorcal={derisked_sorcal:.4f}) in {en-st:.2f} seconds")
        
        # Derisked MAD approach (if enabled)
        if use_mad_optimization:
            logger.write("================\nDerisked MAD Optimization\n================")
            st = time.time()
            derisked_mad_results, _ = run_backtest(test_df, tickers_file=tickers_file, initial_capital=10000, rebalance_frequency=rebalance_frequency, cluster_string=cluster_string, mode="backward", derisk_mode=True, 
                                            derisk_threshold=derisk_threshold, derisk_duration=derisk_duration, derisk_lookahead=derisk_lookahead,
                                            bond_years=10, epsilon=20, k=k, objective=objective, min_alloc=min_alloc, max_alloc=max_alloc, debug=True, rfr=rfr, exponential=exponential, logger=logger, write_alloc=True, fee_type="per_share", fee_pct=0.0, fee_per_share=0.0035,
                                            percentile_threshold=percentile_threshold, lookback_days=lookback_days, lower_percentile_threshold=lower_percentile_threshold,
                                            upper_percentile_threshold=upper_percentile_threshold, hedge=hedge,
                                            fng_threshold_high=fng_threshold_high, fng_threshold_low=fng_threshold_low, change_in_fng_threshold=change_in_fng_threshold,
                                            set_all_cash_range=set_all_cash_range, set_all_cash_change=set_all_cash_change, use_mad_optimization=True, use_hedge_optimization=False)
            
            derisked_mad_ann_return, derisked_mad_vol, derisked_mad_mdd, derisked_mad_downside_deviation, derisked_mad_metric, [derisked_mad_sr, derisked_mad_sortino, derisked_mad_calmar, derisked_mad_sorcal] = compute_metrics(derisked_mad_results, objective=objective, rfr=rfr)
            en = time.time()
            logger.write(f"Derisked MAD optimization performance with k={k}: Annualized Return={derisked_mad_ann_return:.2f}%, Vol={derisked_mad_vol:.4f}, MDD={derisked_mad_mdd:.4f} (Sharpe={derisked_mad_sr:.4f}, Sortino={derisked_mad_sortino:.4f}, Calmar={derisked_mad_calmar:.4f}, Sorcal={derisked_mad_sorcal:.4f}) in {en-st:.2f} seconds")
        
        # Derisked Hedge approach (if enabled)
        if use_hedge_optimization:
            logger.write("================\nDerisked Hedge (Multiplicative Weights) Optimization\n================")
            st = time.time()
            derisked_hedge_results, _ = run_backtest(test_df, tickers_file=tickers_file, initial_capital=10000, rebalance_frequency=rebalance_frequency, cluster_string=cluster_string, mode="backward", derisk_mode=True, 
                                            derisk_threshold=derisk_threshold, derisk_duration=derisk_duration, derisk_lookahead=derisk_lookahead,
                                            bond_years=10, epsilon=20, k=k, objective=objective, min_alloc=min_alloc, max_alloc=max_alloc, debug=True, rfr=rfr, exponential=exponential, logger=logger, write_alloc=True, fee_type="per_share", fee_pct=0.0, fee_per_share=0.0035,
                                            percentile_threshold=percentile_threshold, lookback_days=lookback_days, lower_percentile_threshold=lower_percentile_threshold,
                                            upper_percentile_threshold=upper_percentile_threshold, hedge=hedge,
                                            fng_threshold_high=fng_threshold_high, fng_threshold_low=fng_threshold_low, change_in_fng_threshold=change_in_fng_threshold,
                                            set_all_cash_range=set_all_cash_range, set_all_cash_change=set_all_cash_change, use_mad_optimization=False, use_hedge_optimization=True)
            
            derisked_hedge_ann_return, derisked_hedge_vol, derisked_hedge_mdd, derisked_hedge_downside_deviation, derisked_hedge_metric, [derisked_hedge_sr, derisked_hedge_sortino, derisked_hedge_calmar, derisked_hedge_sorcal] = compute_metrics(derisked_hedge_results, objective=objective, rfr=rfr)
            en = time.time()
            logger.write(f"Derisked Hedge optimization performance with k={k}: Annualized Return={derisked_hedge_ann_return:.2f}%, Vol={derisked_hedge_vol:.4f}, MDD={derisked_hedge_mdd:.4f} (Sharpe={derisked_hedge_sr:.4f}, Sortino={derisked_hedge_sortino:.4f}, Calmar={derisked_hedge_calmar:.4f}, Sorcal={derisked_hedge_sorcal:.4f}) in {en-st:.2f} seconds")

    # logger.write("================\nBaseline\n================")
    # baseline_results, _ = run_backtest(test_df, initial_capital=10000, rebalance_frequency=REBALANCE_FREQUENCY,
    #                                    years=10, epsilon=0, k=30, objective="return", rfr=rfr)

    labels = []
    
    spx_label = "SPX Buy & Hold"
    gold_label = "Gold Buy & Hold"
    bonds_label = "Bonds Buy & Hold"
    regret_label = "Regret Learning"
    mad_label = "MAD Optimization"
    hedge_label = "Hedge"
    if test_derisk:
        derisked_label = "Derisked Portfolio"
        if use_mad_optimization:
            derisked_mad_label = "Derisked MAD Optimization"
        if use_hedge_optimization:
            derisked_hedge_label = "Derisked Hedge Optimization"
    # Compare performance
    all_results = {}
    
    all_results[spx_label] = spx_results
    all_results[gold_label] = gold_results
    all_results[bonds_label] = bonds_results
    all_results[regret_label] = regret_results
    if use_mad_optimization and mad_results is not None:
        all_results[mad_label] = mad_results
    if use_hedge_optimization and hedge_results is not None:
        all_results[hedge_label] = hedge_results
    if test_derisk:
        all_results[derisked_label] = derisked_results
        if use_mad_optimization and derisked_mad_results is not None:
            all_results[derisked_mad_label] = derisked_mad_results
        if use_hedge_optimization and derisked_hedge_results is not None:
            all_results[derisked_hedge_label] = derisked_hedge_results

    plot_performance(all_results, logger=logger, rebalance_dates=regret_rebalance_dates)
    plot_metrics(all_results, rfr=rfr, logger=logger)

    if write_csv:
        save_results_json(vars_dict, all_results, logger)

    logger.close()
    return logger.get_log_dir()

if __name__ == "__main__":
    # with open("tests4.json", "r") as f:
    #     tests = json.load(f)

    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]):
            with open(sys.argv[1], "r") as f:
                tests = json.load(f)
        else:
            raise FileNotFoundError(f"File {sys.argv[1]} not found")

    print(f"Running {len(tests)} tests...")

    for test in tests:
        # run_complete_backtest(**test)
        try:
            log_dir = run_complete_backtest(**test)
            print(f"Logged to {log_dir}")
        except Exception as e:
            traceback.print_exc()
            filename = f"{test['rebalance_frequency']}_{test['objective']}_{test['k']}_{test['cluster_string']}"
            with open(f"logs/{filename}.log", "w") as f:
                f.write(f"Error: {str(e)}")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from metrics import annualized_return, volatility, max_drawdown, sharpe_ratio, downside_deviation, sortino_ratio, calmar_ratio, sorcal_ratio
import json
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("WARNING: cvxpy not available. MAD optimization will be disabled.")

def get_divergence(previous_policy, scenario):
    #softmax over both and get kl div
    previous_policy_values = np.array(list(previous_policy.values()))
    scenario_values = np.array(list(scenario.values()))
    
    previous_policy_softmax = np.exp(previous_policy_values) / np.sum(np.exp(previous_policy_values))
    scenario_softmax = np.exp(scenario_values) / np.sum(np.exp(scenario_values))
    return np.sum(previous_policy_softmax * np.log(previous_policy_softmax / scenario_softmax))

def normalize_scenario(scenario):
    total = sum(scenario.values())
    if total > 0:
        return {k: v/total for k, v in scenario.items()}
    else:
        return {k: 0.0 for k in scenario.keys()}

def generate_scenarios(baseline_allocation, epsilon, uniform_adjustment=True, min_alloc=25, max_alloc=65, k=60, cluster_assets=None):
    """
    Python translation of generate_scenarios_cpp
    Generates portfolio allocation scenarios based on baseline allocation
    """
    assets = list(baseline_allocation.keys())
    cluster_set = set(cluster_assets) if cluster_assets else set()
    
    # Create cluster-aware baseline allocation
    working_baseline = {}
    assets_to_vary = []
    
    if not cluster_assets:
        # If no cluster specified, use original behavior
        working_baseline = baseline_allocation.copy()
        assets_to_vary = assets
    else:
        # Create new baseline with non-cluster assets set to 0
        cluster_sum = 0.0
        for asset_name, weight in baseline_allocation.items():
            if asset_name in cluster_set or asset_name == "Cash" or asset_name == "Bonds":
                working_baseline[asset_name] = weight
                cluster_sum += weight
                assets_to_vary.append(asset_name)
            else:
                working_baseline[asset_name] = 0.0
        
        # Renormalize cluster assets to sum to 1.0
        if cluster_sum > 0.0:
            for asset in assets_to_vary:
                working_baseline[asset] = working_baseline[asset] / cluster_sum
    
    scenarios = []
    scenario_keys = {}  # For deduplication - use dict to maintain order
    
    print(f"Generating scenarios with k={k} epsilon={epsilon}, uniform_adjustment={uniform_adjustment}, " +
          f"min_alloc={min_alloc}, max_alloc={max_alloc}, cluster_size={len(cluster_assets) if cluster_assets else 0}")
    
    # Helper function to round to 3 decimal places
    def round3(x):
        return round(x, 3)
    
    # Helper function to convert scenario to string for deduplication
    def scenario_to_string(scenario):
        return ';'.join(f"{k}:{v}" for k, v in sorted(scenario.items()))
    
    for asset in assets_to_vary:
        base = working_baseline[asset]
        start = min_alloc
        end = max_alloc
        invalidated = 0
        
        if epsilon == 0:
            start = int(base * 100)
            end = start
        
        for i in range(start, end + 1):
            new_allocation_asset = i / 100.0
            diff = new_allocation_asset - base
            
            # Create scenario candidate starting with working baseline
            # print(f"Changing {asset} from {base} to {new_allocation_asset}")
            scenario_candidate = working_baseline.copy()
            scenario_candidate[asset] = round3(new_allocation_asset)
            
            # Determine which assets can be adjusted to compensate for the change
            adjustable_assets = [a for a in assets_to_vary if a != asset]
            
            if uniform_adjustment:
                # Distribute change equally among other assets
                n = len(adjustable_assets)
                # print(f"Adjustable assets: {adjustable_assets}")
                # print(f"Diff for each asset: {diff/n}")
                if n > 0:
                    for oa in adjustable_assets:
                        adjusted = scenario_candidate[oa] - (diff / n)
                        # print(f"Adjusting {oa} from {scenario_candidate[oa]} to {adjusted}")
                        scenario_candidate[oa] = round3(max(adjusted, 0.0))
            else:
                # Non-uniform adjustment: try to use Cash first
                if "Cash" in adjustable_assets:
                    cash = scenario_candidate["Cash"]
                    if cash - diff < 0:
                        diff -= cash
                        scenario_candidate["Cash"] = 0.0
                        non_cash_adjustable = [a for a in adjustable_assets if a != "Cash"]
                        n = len(non_cash_adjustable)
                        if n > 0:
                            for oa in non_cash_adjustable:
                                adjusted = scenario_candidate[oa] - diff / n
                                scenario_candidate[oa] = round3(max(adjusted, 0.0))
                    else:
                        new_cash = scenario_candidate["Cash"] - diff
                        scenario_candidate["Cash"] = round3(max(new_cash, 0.0))
            
            # Iterative uniform adjustment to ensure sum = 1.0 within tolerance
            tolerance = 1e-3
            max_iterations = 100
            iteration = 0
            
            while iteration < max_iterations:
                scenario_sum = sum(scenario_candidate.values())
                
                if abs(scenario_sum - 1.0) <= tolerance:
                    break  # Within tolerance, we're good
                
                adjustment_needed = 1.0 - scenario_sum
                
                # Get all non-zero assets that can be adjusted
                adjustable_non_zero = [a for a in assets if scenario_candidate.get(a, 0) > 0]
                
                if not adjustable_non_zero:
                    break  # No assets to adjust
                
                unit_adjustment = adjustment_needed / len(adjustable_non_zero)
                
                # Apply uniform adjustment
                for asset_name in adjustable_non_zero:
                    new_value = scenario_candidate[asset_name] + unit_adjustment
                    scenario_candidate[asset_name] = round3(max(new_value, 0.0))
                
                iteration += 1
                
            
            # Final validation
            # scenario_candidate = normalize_scenario(scenario_candidate)
            total = sum(scenario_candidate.values())
            if abs(total - 1.0) > tolerance:
                invalidated += 1
                
                # print(f"Invalidated scenario for {asset}: {normalize_scenario(scenario_candidate)} with total {sum(normalize_scenario(scenario_candidate).values())}")
                # print(f"Scenario candidate: {scenario_candidate} with total {total}")
                # exit()
                continue
            
            # Check all weights are valid
            valid = all(0.0 <= v <= 1.0 for v in scenario_candidate.values())
            if not valid:
                continue
            
            # Deduplication check
            key = scenario_to_string(scenario_candidate)
            if key not in scenario_keys:
                scenario_keys[key] = True
                scenarios.append(scenario_candidate)
                
        # print(f"Invalidated {invalidated} scenarios for {asset}")
    
    print(f"Generated {len(scenarios)} scenarios.")
    return scenarios

def scenario_performance(df, assets_ret, asset_names, asset_indices, scenario, objective, exponential=False, steepness=1, rfr=None, regularize=False, reg_factor=0.5, previous_policy=None, cluster=None):
    df_window = df
    window_assets = assets_ret
    
    # Handle empty window
    if window_assets.shape[0] == 0:
        return None
    
    #delete all in asset names and asset indices that are not in scenario
    asset_names = [asset for asset in asset_names if asset in scenario]
    asset_indices = {asset: asset_indices[asset] for asset in asset_names}
    
    try:
        cols_to_check = [asset_indices[asset] for asset in asset_names]
        col_names = [asset for asset in asset_names if scenario.get(asset, 0) > 0 and asset not in ["Cash", "CPI"]]
    except:
        print(f"Scenario: {scenario}")
        print(f"Asset names: {asset_names}")
        print(f"Asset indices: {asset_indices}")
        raise ValueError("Error in scenario performance")
    
    if cols_to_check:
        if -1 in df_window[:, cols_to_check].flatten():
            return None
        
    if cluster is not None:
        for col in col_names:
            if col not in cluster:
                return None
    
    non_cash_indices = [asset_indices[asset] for asset in asset_names]
    weight_vector = np.array([scenario[asset] for asset in asset_names])

    window_array = window_assets[:, non_cash_indices]
    
    portfolio_ret = np.dot(window_array, weight_vector)
    
    # Handle single-day case
    if len(portfolio_ret) == 1:
        # For a single day, the return is just the portfolio return for that day
        single_day_return = portfolio_ret[0]
        
        if objective == "return":
            # Annualize the single day return
            ann = ((1 + single_day_return) ** 252 - 1) * 100
            return ann
        elif objective == "volatility":
            # Can't calculate volatility from single day
            return 0.0
        elif objective == "drawdown":
            # No drawdown possible in single day if return is positive
            return 0.0 if single_day_return >= 0 else -abs(single_day_return)
        elif objective in ["sharpe", "sortino", "calmar"]:
            # These ratios need multiple days
            return 0.0
        else:
            return single_day_return
    
    # Standard calculation for multiple days
    daily_values = np.cumprod(1 + portfolio_ret)
    
    if exponential:
        n = len(portfolio_ret)
        weights = np.exp(np.linspace(0, -steepness, n))
        weights /= weights.sum()
        log_returns = np.log(1 + portfolio_ret)
        weighted_log = np.sum(log_returns * weights)
        final_return = np.exp(weighted_log) - 1.0
    else:
        final_return = daily_values[-1] - 1.0    
    
    if objective == "return":
        final_metric = annualized_return(daily_values)
    elif objective == "volatility":
        vol = volatility(portfolio_ret)
        final_metric = -vol
    elif objective == "drawdown":
        mdd = max_drawdown(daily_values)
        final_metric = -mdd
    elif objective == "calmar":
        calmar = calmar_ratio(daily_values)
        final_metric = calmar
    elif objective == "sharpe":
        sr = sharpe_ratio(daily_values, portfolio_ret, rfr=rfr)
        final_metric = sr
    elif objective == "sortino":
        sortino = sortino_ratio(daily_values, portfolio_ret, rfr=rfr)
        final_metric = sortino
    elif objective == "sorcal":
        sorcal = sorcal_ratio(daily_values, portfolio_ret, rfr=rfr)
        final_metric = sorcal
    elif objective == "final_return":
        final_metric = final_return
    else:
        raise ValueError("Invalid objective")
    
    if regularize:
        if previous_policy is None:
            raise ValueError("Previous policy is required for regularization")
        
        policy_diff = get_divergence(previous_policy, scenario)
        final_metric = final_metric - (reg_factor * policy_diff)
        
        print(f"Policy diff: {policy_diff}")

    return final_metric

def optimize_mad_portfolio(returns_data, asset_names, cluster=None, blacklist=None, min_weight=0.0, max_weight=1.0):
    """
    Optimize portfolio using Mean Absolute Deviation (MAD) criterion.
    
    Args:
        returns_data: numpy array of asset returns (T x N) where T is time periods, N is assets
        asset_names: list of asset names corresponding to columns in returns_data
        cluster: list of assets to consider (if None, consider all assets)
        blacklist: set of assets to exclude
        min_weight: minimum weight for each asset
        max_weight: maximum weight for each asset
    
    Returns:
        dict: optimal portfolio weights {asset_name: weight}
    """
    if not CVXPY_AVAILABLE:
        raise ImportError("cvxpy is required for MAD optimization")
    
    # Filter assets based on cluster and blacklist
    valid_assets = []
    valid_indices = []
    
    for i, asset in enumerate(asset_names):
        # Skip if blacklisted
        if blacklist is not None and asset in blacklist:
            continue
            
        # Skip if not in cluster (except for Cash and Bonds)
        if cluster is not None and asset not in cluster and asset not in ['Cash', 'Bonds']:
            continue
            
        valid_assets.append(asset)
        valid_indices.append(i)
    
    if len(valid_assets) == 0:
        # Fallback to cash if no valid assets
        return {'Cash': 1.0}
    
    # Extract returns for valid assets
    R = returns_data[:, valid_indices]
    n_assets = len(valid_assets)
    
    # Check for missing data (-1 values)
    if np.any(R == -1):
        print("WARNING: Found missing data (-1) in returns, falling back to equal weight")
        equal_weight = 1.0 / n_assets
        return {asset: equal_weight for asset in valid_assets}
    
    # CVXPY optimization variables
    x = cp.Variable(n_assets, nonneg=True)
    
    # Portfolio returns for each time period
    portfolio_returns = R @ x
    
    # Expected return
    expected_return = cp.sum(R @ x) / R.shape[0]
    
    # Mean Absolute Deviation
    mad = cp.sum(cp.abs(portfolio_returns - expected_return)) / R.shape[0]
    
    # Constraints
    constraints = [
        cp.sum(x) == 1,  # weights sum to 1
        x >= min_weight,  # minimum weight constraint
        x <= max_weight   # maximum weight constraint
    ]
    
    # Optimization problem
    prob = cp.Problem(cp.Minimize(mad), constraints)
    
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        
        if prob.status not in ["infeasible", "unbounded"]:
            optimal_weights = x.value
            
            # Create result dictionary
            result = {}
            for i, asset in enumerate(valid_assets):
                result[asset] = max(0.0, optimal_weights[i])  # Ensure non-negative
            
            # Normalize weights to sum to 1
            total_weight = sum(result.values())
            if total_weight > 0:
                result = {asset: weight / total_weight for asset, weight in result.items()}
            else:
                # Fallback to equal weight
                equal_weight = 1.0 / len(valid_assets)
                result = {asset: equal_weight for asset in valid_assets}
            
            # Add zero weights for assets not in the optimization
            for asset in asset_names:
                if asset not in result:
                    result[asset] = 0.0
            
            print(f"MAD optimization successful. MAD value: {mad.value:.6f}")
            return result
        else:
            print(f"MAD optimization failed with status: {prob.status}")
    except Exception as e:
        print(f"MAD optimization error: {str(e)}")
    
    # Fallback to equal weight among valid assets
    print("Falling back to equal weight portfolio")
    equal_weight = 1.0 / len(valid_assets)
    result = {asset: equal_weight for asset in valid_assets}
    
    # Add zero weights for assets not in the optimization
    for asset in asset_names:
        if asset not in result:
            result[asset] = 0.0
            
    return result

def pick_best_scenario(df, current_allocation, epsilon, start_date, k, objective, ret_full, mode="backward", rebalance_dates=None, rebalance_frequency=None, 
                       min_alloc=25, max_alloc=65, rfr=None, exponential=False, regularize=False, reg_factor=0.5, cluster=None, blacklist=None,
                       percentile_threshold=None, lookback_days=None, lower_percentile_threshold=None, upper_percentile_threshold=None, use_mad_optimization=False):
    print(f"Picking best scenario for start_date: {start_date}")
    if cluster is not None:
        print(f"Cluster length: {len(cluster)}")
    if blacklist is not None:
        print(f"Blacklisted assets: {blacklist}")
    
    # Early return for MAD optimization
    if use_mad_optimization:
        print("Using MAD optimization for portfolio allocation")
        
        if not CVXPY_AVAILABLE:
            print("ERROR: cvxpy not available, falling back to current allocation")
            return current_allocation
            
        # Set up the lookback window for MAD optimization
        if start_date not in df.index:
            later_dates = df.index[df.index > start_date]
            earlier_dates = df.index[df.index < start_date]
            if len(later_dates) > 0 and len(earlier_dates) > 0:
                start_date = later_dates.min()
            else:
                return current_allocation
        
        # Get the lookback window (past k days)
        pos = df.index.get_loc(start_date)
        end_idx = pos - 1
        start_idx = max(0, end_idx - k + 1)
        
        if start_idx < 0 or start_idx > end_idx:
            print(f"WARNING: Invalid window for MAD optimization - start_idx ({start_idx}) > end_idx ({end_idx})")
            return current_allocation
        
        # Extract returns data for the lookback period
        ret_window = ret_full.iloc[start_idx:end_idx+1].drop(columns=["CPI"], axis=1, errors='ignore')
        ret_window["Cash"] = [0.0] * ret_window.shape[0]  # Cash has 0 return
        
        asset_names = list(ret_window.columns)
        returns_data = ret_window.to_numpy()
        
        print(f"MAD optimization window: {ret_window.shape[0]} days, {len(asset_names)} assets")
        
        # Apply min/max allocation constraints
        min_weight = min_alloc / 100.0  # Convert percentage to decimal
        max_weight = max_alloc / 100.0  # Convert percentage to decimal
        
        try:
            mad_allocation = optimize_mad_portfolio(
                returns_data, 
                asset_names, 
                cluster=cluster, 
                blacklist=blacklist
            )
            
            print(f"MAD optimization result: {[(k, v) for k, v in mad_allocation.items() if v > 0.001]}")
            return mad_allocation
            
        except Exception as e:
            print(f"MAD optimization failed: {str(e)}")
            return current_allocation
    
    if regularize and current_allocation is None:
        raise ValueError("Current allocation is required for regularization")
    
    if start_date not in df.index:
        # Find the next available date after start_date, but not at the boundaries
        later_dates = df.index[df.index > start_date]
        earlier_dates = df.index[df.index < start_date]
        # Only proceed if not at the boundaries
        if len(later_dates) > 0 and len(earlier_dates) > 0:
            # Use the next available date after start_date
            start_date = later_dates.min()
        else:
            return current_allocation
    
    print(f"df shape: {df.shape}")

    if mode == "backward":
        pos = df.index.get_loc(start_date)
        end_idx = pos - 1
        start_idx = end_idx - k + 1
    else:
        if rebalance_dates is None:
            raise ValueError("Rebalance dates must be provided for forward mode")
    
        # For all forward modes (including daily), look ahead to the next rebalance period
        pos = rebalance_dates.get_loc(start_date)
        
        # Get the period from today to next rebalance
        start_idx = df.index.get_loc(start_date) + 1  # Start from tomorrow
        
        try:
            if (pos+1) >= len(rebalance_dates):
                # Last rebalance period - go to end of data
                end_date = df.index[-1]
            else:
                # Go to the next rebalance date
                end_date = rebalance_dates[pos+1]
                # But we want the day before the next rebalance
                end_idx_temp = df.index.get_loc(end_date)
                if end_idx_temp > 0:
                    end_date = df.index[end_idx_temp]
        except:
            print(f"Error getting end date for {start_date}")
            return current_allocation
            
        end_idx = df.index.get_loc(end_date)
        
        # Debug logging for forward mode
        print(f"Forward {rebalance_frequency}: start_date={start_date}, looking ahead from {df.index[start_idx] if start_idx < len(df) else 'end'} to {end_date}")
        print(f"  Window size: {end_idx - start_idx + 1} days")
        
        # For all forward modes, we evaluate scenarios over the future period
        # This finds the allocation that performs best while respecting min/max constraints

    if start_idx < 0:
        print(f"WARNING: start_idx ({start_idx}) is less than 0")
        return current_allocation
    
    # Add check for empty or invalid window
    if start_idx > end_idx:
        print(f"WARNING: Invalid window - start_idx ({start_idx}) > end_idx ({end_idx})")
        return current_allocation
    
    # Check if we're at the end of data
    if start_idx >= len(df):
        print(f"WARNING: start_idx ({start_idx}) is at or beyond end of data")
        return current_allocation
    
    st = time.time()

    # Filter out blacklisted assets from current allocation before generating scenarios
    filtered_allocation = current_allocation.copy()
    if blacklist is not None:
        for asset in blacklist:
            if asset in filtered_allocation:
                filtered_allocation[asset] = 0.0
                print(f"Removed blacklisted asset: {asset}")
        # Renormalize the allocation after removing blacklisted assets
        # total_remaining = sum(filtered_allocation.values())
        # if total_remaining > 0:
        #     filtered_allocation = {k: v/total_remaining for k, v in filtered_allocation.items()}
        # else:
        #     # If all assets are blacklisted, default to cash
        #     filtered_allocation = {k: 0.0 for k in filtered_allocation.keys()}
        #     filtered_allocation['Cash'] = 1.0

    # Percentile performance filtering
    assets_to_keep = None  # Initialize outside the if blocks
    
    if percentile_threshold is not None and lookback_days is not None:
        print(f"Applying percentile filtering: threshold={percentile_threshold}, lookback_days={lookback_days}")
        
        # Calculate lookback period
        current_pos = df.index.get_loc(start_date)
        lookback_start_idx = max(0, current_pos - lookback_days)
        
        if current_pos >= lookback_start_idx:
            # Get price data for lookback period
            lookback_df = df.iloc[lookback_start_idx:current_pos+1]
            
            # Calculate returns for each asset over the lookback period
            asset_returns = {}
            valid_assets = []
            
            # Check ALL assets in the dataframe, not just currently allocated ones
            for asset in cluster:
                if asset in ['Cash', 'CPI']:
                    continue
                    
                # Check if asset has valid prices (not -1) during lookback period
                if asset in lookback_df.columns:
                    prices = lookback_df[asset].values
                    print(prices)
                    if len(prices) > 1 and not np.any(prices == -1):
                        # Calculate simple return
                        asset_return = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
                        asset_returns[asset] = asset_return
                        valid_assets.append(asset)
            
            if len(valid_assets) > 0:
                # Calculate percentile threshold
                returns_array = np.array(list(asset_returns.values()))
                threshold_value = np.percentile(returns_array, percentile_threshold)
                
                if lower_percentile_threshold is not None:
                    lower_threshold_value = np.percentile(returns_array, lower_percentile_threshold)
                if upper_percentile_threshold is not None:
                    upper_threshold_value = np.percentile(returns_array, upper_percentile_threshold)
                
                # Find top performers
                top_performers = []
                for asset, ret in asset_returns.items():
                    if lower_percentile_threshold is None and upper_percentile_threshold is None:
                        if ret >= threshold_value:
                            top_performers.append(asset)
                    elif lower_percentile_threshold is not None and upper_percentile_threshold is not None:
                        if ret >= lower_threshold_value and ret <= upper_threshold_value:
                            top_performers.append(asset)
                    else:
                        raise ValueError("Lower and upper percentile thresholds must both be provided or neither")
                
                # Get currently invested assets (non-zero allocation)
                currently_invested = [asset for asset in filtered_allocation.keys() 
                                    if filtered_allocation[asset] > 0 and asset != 'Cash']
                
                # Union of currently invested and top performers
                assets_to_keep = list(set(currently_invested + top_performers))
                
                # Always include Cash
                if 'Cash' not in assets_to_keep:
                    assets_to_keep.append('Cash')
                # # Include Bonds if it exists
                # if 'Bonds' in df.columns and 'Bonds' not in assets_to_keep:
                #     assets_to_keep.append('Bonds')
                if 'Bonds' in assets_to_keep:
                    assets_to_keep.remove('Bonds')
                
                print(f"Currently invested assets: {len(currently_invested)}")
                if lower_percentile_threshold is None and upper_percentile_threshold is None:
                    print(f"Top performing assets (top {(100-percentile_threshold):.0f}%): {len(top_performers)}")
                elif lower_percentile_threshold is not None and upper_percentile_threshold is not None:
                    print(f"Top performing assets (between {lower_percentile_threshold:.0f}% and {upper_percentile_threshold:.0f}%): {len(top_performers)}")
                else:
                    raise ValueError("Lower and upper percentile thresholds must both be provided or neither")
                
                print(f"Total assets to keep (union + Cash/Bonds): {len(assets_to_keep)}")
                
                if lower_percentile_threshold is None and upper_percentile_threshold is None:
                    print(f"Threshold return value: {threshold_value:.4f}")
                elif lower_percentile_threshold is not None and upper_percentile_threshold is not None:
                    print(f"Lower threshold return value: {lower_threshold_value:.4f}")
                    print(f"Upper threshold return value: {upper_threshold_value:.4f}")
                else:
                    raise ValueError("Lower and upper percentile thresholds must both be provided or neither")
                
                # Update filtered_allocation to exclude assets not in our keep list
                for asset in list(filtered_allocation.keys()):
                    if asset not in assets_to_keep:
                        del filtered_allocation[asset]
                        
                # Note: We don't renormalize here because we want the scenario generator
                # to explore different allocations among the allowed assets
                
            else:
                print("No valid assets found for percentile filtering")
        else:
            print(f"Not enough history for percentile filtering (need {lookback_days} days)")
    
    
    # equal_share = 1.0 / len(filtered_allocation)
    # for asset in filtered_allocation:
    #     filtered_allocation[asset] = equal_share
    
    print(f"Total Cluster Population: {len(assets_to_keep)}")
    print(f"assets to keep: {assets_to_keep}")
    print(f"filtered allocation: {filtered_allocation}")

    scenarios = generate_scenarios(filtered_allocation, epsilon, True, min_alloc, max_alloc, k, assets_to_keep)
    
    # # Filter scenarios to remove any that allocate to blacklisted assets
    # if blacklist is not None:
    #     filtered_scenarios = []
    #     for scenario in scenarios:
    #         has_blacklisted = any(scenario.get(asset, 0) > 0 for asset in blacklist)
    #         if not has_blacklisted:
    #             filtered_scenarios.append(scenario)
    #     scenarios = filtered_scenarios
    
    print(f"Scenarios generated: {len(scenarios)} in {time.time()-st:.2f} seconds")
    # with open("scenarios.json", "w") as f:
    #     json.dump(scenarios, f, indent=4)
    # exit()

    df_window = df.iloc[start_idx:end_idx+1].drop(columns=["CPI"], axis=1)
    ret_full = ret_full.iloc[start_idx:end_idx+1].drop(columns=["CPI"], axis=1)
    ret_full["Cash"] = [0.0] * ret_full.shape[0]
    
    if mode == "forward":
        print(f"df_window shape: {df_window.shape}")
        print(f"ret_full shape: {ret_full.shape}")

    asset_names = list(ret_full.columns)

    asset_names_filtered = [asset for asset in asset_names if asset not in ["CPI", "Cash"]]
    asset_indices = {asset: asset_names.index(asset) for asset in asset_names if asset not in ["CPI", "Cash"]}

    df_window = df_window.to_numpy()
    ret_full = ret_full.to_numpy()

    best_score = None
    best_scenario = current_allocation
    
    if ret_full.shape[0] == 0:
        print(f"Ret full shape: {ret_full.shape} is 0")
        return current_allocation

    # scores = batch_evaluate_scenarios(ret_full, df_window, asset_names, scenarios, objective, exponential, 1.0, rfr)
    # # print(scores)
    # # exit()

    # for i, score in enumerate(scores):
    #     if not math.isnan(score):
    #         if (best_score is None) or (score > best_score):
    #             best_score = score
    #             best_scenario = scenarios[i]

    scores = [scenario_performance(df_window, ret_full, asset_names_filtered, asset_indices, s, objective, exponential, 1.0, rfr, regularize, reg_factor, current_allocation, cluster) for s in scenarios]
    
    # Debug: Count None scores
    none_count = sum(1 for s in scores if s is None)
    if none_count > 0:
        print(f"WARNING: {none_count}/{len(scores)} scenarios returned None")
        if none_count == len(scores):
            print("All scenarios are invalid! Debugging...")
            print(f"Window shape: {ret_full.shape}")
            print(f"Cluster assets: {cluster}")
            print(f"Asset names filtered: {asset_names_filtered}")
            # Check a few scenarios to see why they're failing
            for i, scenario in enumerate(scenarios[:3]):
                allocated_assets = [asset for asset, weight in scenario.items() if weight > 0]
                print(f"Sample scenario {i}: {allocated_assets}")
                if cluster is not None:
                    invalid_assets = [a for a in allocated_assets if a not in cluster and a not in ['Cash', 'Bonds']]
                    if invalid_assets:
                        print(f"  Invalid assets not in cluster: {invalid_assets}")
    
    scores = [s if s is not None else -math.inf for s in scores]

    best_score = max(scores)
    
    # Safety check for case where all scenarios are invalid
    if best_score == -math.inf:
        print(f"WARNING: All scenarios are invalid, falling back to current allocation")
        print(f"Number of scenarios: {len(scenarios)}")
        print(f"Number of None scores: {sum(1 for s in scores if s == -math.inf)}")
        return current_allocation
    
    # Use original tie-breaking approach (first occurrence)
    best_scenario = scenarios[scores.index(best_score)]
    
    # Debug logging for chosen scenario
    if mode == "forward":
        print(f"  Best score: {best_score}")
        print(f"  Best scenario allocation: {[(k,v) for k,v in best_scenario.items() if v > 0]}")
        # Check if we're defaulting to cash
        if best_scenario.get('Cash', 0) > 0.99:
            print(f"  WARNING: Mostly/all in cash!")
            print(f"  Number of valid scores: {sum(1 for s in scores if s != -math.inf)}")

    print(f"Time taken for best of {len(scenarios)} scenarios: {time.time()-st:.2f} seconds")

    return best_scenario


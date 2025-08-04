import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import time

def annualized_return(nominal):
    total_days = len(nominal)
    return ((nominal[-1] / nominal[0]) ** (252 / total_days) - 1) * 100

def volatility(returns):
    return np.std(returns, ddof=1) * np.sqrt(252)

def sharpe_ratio(nominal, returns, rfr=0.0):
    return ((annualized_return(nominal) - rfr) / volatility(returns)) / 100

def downside_deviation(returns):
    neg_returns = returns[returns < 0]
    if neg_returns.size < 2:
        return 0.0
    return np.std(neg_returns, ddof=1) * np.sqrt(252)

def max_drawdown(nominal):
    cum_max = np.maximum.accumulate(nominal)
    drawdown = (nominal - cum_max) / cum_max
    return np.abs(np.min(drawdown))

def sortino_ratio(nominal, returns, rfr=0.0):
    dd = downside_deviation(returns)
    return (annualized_return(nominal) - rfr) / dd if dd != 0 else 0.0

def calmar_ratio(nominal):
    if max_drawdown(nominal) == 0:
        return 0.0
    return annualized_return(nominal) / max_drawdown(nominal)

def sorcal_ratio(nominal, returns, rfr=0.0):
    sortino = sortino_ratio(nominal, returns, rfr=rfr)
    calmar = calmar_ratio(nominal)
    if calmar == 0:
        return sortino
    return sortino * calmar

def compute_metrics(results, objective="custom", rfr=None):
    nominal = results['Total_Nominal'].dropna()
    returns = nominal.pct_change().dropna().to_numpy()

    nominal = nominal.to_numpy()

    ann = round(annualized_return(nominal),2)

    if rfr is None:
        return ann, 0, 0, 0, ann, [0,0,0,0]
    
    final_return = (nominal[-1] / nominal[0]) - 1.0
    vol = round(volatility(returns), 2)
    mdd = round(max_drawdown(nominal), 2)
    down_deviation = round(downside_deviation(returns), 2)
    
    sr = round(sharpe_ratio(nominal, returns, rfr=rfr), 2)
    sortino = round(sortino_ratio(nominal, returns, rfr=rfr), 2) / 100
    calmar = round(calmar_ratio(nominal), 2) / 100
    sorcal = round(sorcal_ratio(nominal, returns, rfr=rfr), 2) / 100
    
    if objective == "return":
        metric = final_return
    elif objective == "volatility":
        metric = -volatility
    elif objective == "drawdown":
        metric = -abs(max_drawdown)
    elif objective == "calmar":
        metric = calmar
    elif objective == "sharpe":
        metric = sr
    elif objective == "sortino":
        metric = sortino
    elif objective == "sorcal":
        metric = sorcal
    else:
        metric = ann  # default

    all_metrics = [sr, sortino, calmar, sorcal]

    return ann, vol, mdd, down_deviation, metric, all_metrics
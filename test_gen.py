import json
from itertools import product

universes = ["SECTOR-ETFs"]
cluster_strings = ["Z"]
rebalance_frequencies = ["Y", "Q", "M"]
objectives = ["return", "calmar", "sortino"]
ks = [5, 10, 20, 30, 60, 90, 120]
percentile_thresholds = [75, 50, 0]
lookback_days = [10, 30, 50]
fng_thresholds = [
    {"fng_threshold_low": 10, "fng_threshold_high": 90},
    {"fng_threshold_low": 30, "fng_threshold_high": 70},
    {"fng_threshold_low": 20, "fng_threshold_high": 80},
]

ticker_files = {
    "SECTOR-ETFs": "tickers_sector_etfs.json",
    "SPDR500": "tickers_spdr500.json",
}

# Generate all combinations from the ranges you specified
# [0-5, 10-30, 1-10] means all combinations of these ranges
# change_in_fng_thresholds = [[0, 20, 5]]
change_in_fng_thresholds = [None, [0, 20, 5], [0, 10, 5], [0, 5, 2]]
# for range1 in range(0, 1):  # 0-5
#     for range2 in range(15, 26):  # 10-30
#         for range3 in range(1, 6):  # 1-10
#             change_in_fng_thresholds.append([range1, range2, range3])

tests = []

for universe in universes:
    start_date = "2001-01-01" if universe == "SPDR500" else "2011-01-01"
    tickers_file = ticker_files[universe]
    for cluster_string in cluster_strings:
        for rebalance_frequency in rebalance_frequencies:
            for objective in objectives:
                for k in ks:
                    # Hedge logic
                    if cluster_string == "Z":
                        hedge_combos = [(0, "Z")]
                    else:
                        hedge_combos = [(1, "dynamic"), (0, "dynamic")]
                    for hedge, cluster in hedge_combos:
                        for fng in fng_thresholds:
                            for change_in_fng in change_in_fng_thresholds:
                                for percentile_threshold in percentile_thresholds:
                                    for lookback_day in lookback_days:
                                        test = {
                                            "universe": universe,
                                            "start_date": start_date,
                                            "end_date": "2024-12-31",
                                            "tickers_file": tickers_file,
                                            "rebalance_frequency": rebalance_frequency,
                                            "objective": objective,
                                            "k": k,
                                            "min_alloc": 30,
                                            "max_alloc": 80,
                                            "exponential": 0,
                                            "cluster_string": cluster_string,
                                            "test_derisk": 0,
                                            "derisk_threshold": 0.1,
                                            "derisk_duration": 5,
                                            "derisk_lookahead": 15,
                                            "write_csv": 1,
                                            "regularize": 0,
                                            "reg_factor": 1000,
                                            "percentile_threshold": percentile_threshold,
                                            "lookback_days": lookback_day,
                                            "lower_percentile_threshold": None,
                                            "upper_percentile_threshold": None,
                                            "hedge": hedge,
                                            "fng_threshold_high": fng["fng_threshold_high"],
                                            "fng_threshold_low": fng["fng_threshold_low"],
                                            "change_in_fng_threshold": change_in_fng,
                                            "set_all_cash_range": 1,
                                            "set_all_cash_change": 1
                                        }
                                        tests.append(test)

print(f"Generated {len(tests)} test combinations")
with open("tests_generated_2.json", "w") as f:
    json.dump(tests, f, indent=4)
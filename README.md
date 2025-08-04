# No-Regret Portfolio Optimization

This repository contains a backtesting system for portfolio optimization using regret learning algorithms with Fear & Greed Index (FNG) filtering.

## Overview

The system implements a sophisticated portfolio optimization strategy that:
- Uses regret learning to minimize portfolio regret over time
- Incorporates Fear & Greed Index data for market sentiment filtering
- Supports multiple asset universes (S&P 500, Sector ETFs, etc.)
- Includes de-risking mechanisms during high volatility periods
- Provides comprehensive performance analysis and comparison

## Setup

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (for dynamic cluster selection)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd no-regret-paper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root:
```
OPENAI_KEY=your_openai_api_key_here
```

### Data Files Required

Ensure the following data files are present:
- `combined_data3.csv` - S&P 500 historical data
- `combined_stock_data_etfs.csv` - Sector ETF data
- `fng.csv` - Fear & Greed Index historical data
- `sectors.json` - Sector mapping configuration
- `hedge_dict.json` - Hedging relationships
- `tickers_sector_etfs.json` - ETF ticker information

## Usage

### Running Tests

The main entry point is `test_nri.py` which runs backtesting scenarios defined in JSON configuration files.

#### Basic Usage
```bash
python test_nri.py tests4.json
```

**Note**: You must provide a test configuration file as a command line argument. The script will not run without it.

#### Test Configuration

Test files are JSON arrays containing configuration objects. Each test configuration supports the following parameters:

- `universe`: Asset universe ("SPDR500" or "SECTOR-ETFs")
- `start_date`/`end_date`: Backtest period
- `rebalance_frequency`: "D", "W", "M", "Q", or "Y"
- `objective`: Optimization objective ("return", "sharpe", "sortino", "calmar")
- `k`: Lookback period for regret calculation
- `min_alloc`/`max_alloc`: Allocation constraints (as percentages)
- `fng_threshold_high`/`fng_threshold_low`: FNG filtering thresholds
- `percentile_threshold`/`lookback_days`: Performance filtering parameters
- `hedge`: Enable/disable hedging (1/0)

### Example Test Run

```bash
python test_nri.py tests4.json
```

This will:
1. Load test configurations from `tests4.json`
2. Run each backtest scenario
3. Generate performance reports and visualizations
4. Save results to the `logs/` directory

### Output

Results are saved in timestamped directories under `logs/` containing:
- `run.log` - Detailed execution log
- `results.json` - Performance metrics
- `vars.json` - Test configuration
- `allocations.csv` - Portfolio allocations over time
- Performance comparison charts (PNG files)

## Test Generation

Use `test_gen.py` to generate comprehensive test suites:

```bash
python test_gen.py
```

This creates `tests_generated_2.json` with systematic parameter combinations.

## Key Components

- `test_nri.py` - Main backtesting runner
- `backtest.py` - Core backtesting logic
- `scenario_utils.py` - Scenario generation and evaluation
- `metrics.py` - Performance metric calculations
- `utils.py` - Utility functions and data processing
- `logger.py` - Logging and result persistence

## Performance Metrics

The system calculates comprehensive performance metrics:
- Annualized returns
- Volatility and downside deviation
- Maximum drawdown
- Sharpe, Sortino, Calmar, and Sorcal ratios
- Correlation analysis

## Advanced Features

### Dynamic Cluster Selection
Uses OpenAI GPT models to dynamically select sector clusters based on Fear & Greed Index patterns.

### De-risking Mechanism
Automatically shifts to defensive allocations during periods of high expected volatility.

### Asset Universe Filtering
- Percentile-based performance filtering
- Blacklist management for delisted assets
- Sector-based clustering with hedging

## Troubleshooting

1. **Missing data files**: Ensure all required CSV and JSON files are present
2. **OpenAI API errors**: Check your API key and rate limits
3. **Memory issues**: Reduce the number of scenarios or test parameters for large backtests
4. **Invalid dates**: Ensure date ranges match available data

## Contributing

When modifying the code:
1. Follow the existing code structure
2. Add comprehensive logging for debugging
3. Test with small parameter sets first
4. Document any new configuration parameters
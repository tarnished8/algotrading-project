# Algorithmic Trading Strategy - Cross-Asset Trend Following

This repository contains my implementation of a cross-asset trend-following algorithmic trading strategy. The project was originally developed as a capstone project for ESILV Algotrading 2024-25.

## Project Overview

The strategy implements a time-series momentum approach across 47 different assets from various asset classes. It uses multiple lookback periods (2 weeks to 2 months) to generate trading signals and calculates positions based on the average of obtained signs for all lookback periods.

## Features

- **Data Processing**: Robust data loading and preprocessing with missing value handling
- **Exploratory Data Analysis**: Comprehensive analysis of asset correlations and availability
- **Trend Analysis**: Implementation of momentum-based signal generation
- **Strategy Implementation**: Complete backtesting framework with position calculation
- **Performance Analysis**: Detailed performance metrics including Sharpe ratios and returns
- **Object-Oriented Design**: Modular, maintainable code structure

## Repository Structure

```
├── src/
│   ├── data/
│   │   └── data_loader.py          # Data loading and preprocessing
│   ├── analysis/
│   │   ├── eda.py                  # Exploratory Data Analysis
│   │   ├── trend_analyzer.py       # Trend and momentum analysis
│   │   └── performance_analyzer.py # Performance metrics and visualization
│   ├── strategy/
│   │   └── trend_following.py      # Main strategy implementation
│   ├── utils/
│   │   └── helpers.py              # Utility functions
│   └── config.py                   # Configuration management
├── data/
│   └── assets.csv                  # Asset price data
├── notebooks/
│   └── original_notebook.ipynb     # Original research notebook
├── tests/
│   └── test_*.py                   # Unit tests
├── main.py                         # Main execution script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd algotrading-strategy
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete strategy analysis:
```bash
python main.py
```

### Custom Configuration

Modify parameters in `src/config.py` to customize:
- Lookback periods
- Asset selection
- Risk management parameters
- Performance metrics

### Individual Components

```python
from src.data.data_loader import DataLoader
from src.strategy.trend_following import TrendFollowingStrategy

# Load data
loader = DataLoader('data/assets.csv')
data = loader.load_and_preprocess()

# Run strategy
strategy = TrendFollowingStrategy()
results = strategy.backtest(data)
```

## Data

The dataset contains daily closing prices for 47 assets across various asset classes from 2001 to 2021. Assets are anonymized with codes like 'x379', 'J428', etc.

### Data Features:
- **Time Period**: 2001-2021 (in-sample data)
- **Frequency**: Daily
- **Assets**: 47 cross-asset instruments
- **Missing Data**: Handled through forward-fill and availability tracking

## Strategy Details

### Signal Generation
- Uses multiple lookback periods (14 to 60 days)
- Calculates momentum signals based on cumulative returns
- Averages signals across all lookback periods
- Normalizes positions to ensure portfolio allocation

### Position Calculation
```python
def compute_positions_trend(returns, periods_long):
    signal = 0 * returns.fillna(0)
    for period in range(periods_long):
        return_sign = np.sign(returns.rolling(window=period).sum())
        signal = signal + return_sign
    allocated = np.abs(signal).sum(axis=1)
    allocated[allocated == 0] = np.nan
    position = signal / allocated.values[:,np.newaxis]
    return position
```

### Risk Management
- Position sizing based on signal strength
- Portfolio-level allocation constraints
- 2-day lag implementation for realistic trading

## Performance Metrics

The strategy is evaluated using:
- **Sharpe Ratio**: Risk-adjusted returns
- **Cumulative Returns**: Total strategy performance
- **Volatility**: Risk measurement
- **Maximum Drawdown**: Downside risk assessment

## Acknowledgments

- ESILV Algotrading Course 2024-25
- Original capstone project framework
- Cross-asset momentum research literature


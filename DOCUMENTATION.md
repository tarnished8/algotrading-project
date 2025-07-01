# Algorithmic Trading Strategy - Technical Documentation

## Overview

This repository implements a cross-asset trend-following algorithmic trading strategy using Object-Oriented Programming principles. The strategy was originally developed as a capstone project for the Algotrading 2024-25 course at ESILV and has been refactored into a modular, maintainable codebase.

## Architecture

### Design Principles

1. **Separation of Concerns**: Each module has a specific responsibility
2. **Object-Oriented Design**: Core functionality is encapsulated in classes
3. **Configuration Management**: All parameters are centralized in configuration files
4. **Testability**: Code is designed to be easily testable with unit tests
5. **Modularity**: Components can be used independently or together

### Module Structure

```
src/
├── config.py                  # Configuration management
├── data/
│   ├── __init__.py
│   └── data_loader.py         # Data loading and preprocessing
├── analysis/
│   ├── __init__.py
│   ├── eda.py                 # Exploratory Data Analysis
│   ├── trend_analyzer.py      # Trend and momentum analysis
│   └── performance_analyzer.py # Performance metrics and visualization
├── strategy/
│   ├── __init__.py
│   └── trend_following.py     # Main strategy implementation
└── utils/
    ├── __init__.py
    └── helpers.py             # Utility functions
```

## Core Components

### 1. Configuration Management (`src/config.py`)

The configuration system uses dataclasses to organize parameters:

- **DataConfig**: Data loading and preprocessing parameters
- **StrategyConfig**: Strategy-specific parameters (lookback periods, position sizing)
- **AnalysisConfig**: Analysis and visualization settings

```python
from src.config import get_config, get_config_by_env

# Get default configuration
config = get_config()

# Get environment-specific configuration
prod_config = get_config_by_env("production")
```

### 2. Data Loading (`src/data/data_loader.py`)

The `DataLoader` class handles:
- CSV file loading with datetime parsing
- Data preprocessing (resampling, missing value handling)
- Log returns calculation
- Data validation

```python
from src.data.data_loader import DataLoader

loader = DataLoader()
data, returns = loader.load_and_preprocess("data/assets.csv")
```

### 3. Exploratory Data Analysis (`src/analysis/eda.py`)

The `ExploratoryDataAnalysis` class provides:
- Asset availability visualization
- Correlation analysis and heatmaps
- Statistical summaries
- Data quality assessment

```python
from src.analysis.eda import ExploratoryDataAnalysis

eda = ExploratoryDataAnalysis()
report = eda.create_comprehensive_report(data, returns)
```

### 4. Trend Analysis (`src/analysis/trend_analyzer.py`)

The `TrendAnalyzer` class implements:
- Momentum signal calculation
- Multi-period trend analysis
- Momentum persistence analysis
- Lookback period optimization

```python
from src.analysis.trend_analyzer import TrendAnalyzer

analyzer = TrendAnalyzer()
signals = analyzer.calculate_multi_period_momentum(returns)
```

### 5. Strategy Implementation (`src/strategy/trend_following.py`)

The `TrendFollowingStrategy` class contains:
- Position calculation based on momentum signals
- Signal generation across multiple lookback periods
- Complete backtesting framework
- Performance metrics calculation

```python
from src.strategy.trend_following import TrendFollowingStrategy

strategy = TrendFollowingStrategy()
results = strategy.backtest(returns)
```

### 6. Performance Analysis (`src/analysis/performance_analyzer.py`)

The `PerformanceAnalyzer` class provides:
- Comprehensive performance metrics
- Risk analysis (VaR, CVaR, drawdowns)
- Performance visualization dashboards
- Benchmark comparisons

```python
from src.analysis.performance_analyzer import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
metrics = analyzer.generate_performance_report(portfolio_returns)
```

## Strategy Details

### Signal Generation

The strategy uses a time-series momentum approach:

1. **Multiple Lookback Periods**: Calculates momentum signals using lookback periods from 14 to 60 days
2. **Signal Averaging**: Averages signals across all lookback periods
3. **Position Normalization**: Normalizes positions to ensure proper portfolio allocation

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

- **Position Sizing**: Positions are normalized by total signal strength
- **Signal Lag**: 2-day lag implementation for realistic trading
- **Position Limits**: Maximum position size constraints
- **Portfolio Constraints**: Total portfolio allocation limits

## Usage Examples

### Basic Usage

```python
# Complete workflow
from src.data.data_loader import DataLoader
from src.strategy.trend_following import TrendFollowingStrategy
from src.analysis.performance_analyzer import PerformanceAnalyzer

# Load data
loader = DataLoader()
data, returns = loader.load_and_preprocess()

# Run strategy
strategy = TrendFollowingStrategy()
results = strategy.backtest(returns)

# Analyze performance
analyzer = PerformanceAnalyzer()
metrics = analyzer.generate_performance_report(results["portfolio_returns"])
```

### Custom Configuration

```python
from src.config import StrategyConfig, AnalysisConfig
from src.strategy.trend_following import TrendFollowingStrategy

# Custom strategy configuration
config = StrategyConfig(
    min_lookback_days=10,
    max_lookback_days=30,
    signal_lag_days=1,
    max_position_size=0.3
)

strategy = TrendFollowingStrategy(config)
```

### Running Analysis Components Separately

```python
# EDA only
from src.analysis.eda import ExploratoryDataAnalysis
eda = ExploratoryDataAnalysis()
eda.plot_asset_availability(data)
eda.plot_correlation_heatmap(returns, year=2018)

# Trend analysis only
from src.analysis.trend_analyzer import TrendAnalyzer
analyzer = TrendAnalyzer()
optimization = analyzer.optimize_lookback_period(returns)
```

## Testing

The codebase includes comprehensive unit tests:

```bash
# Run all tests
python run_tests.py

# Run specific test file
python run_tests.py test_strategy.py

# Run with pytest directly
pytest tests/ -v
```

### Test Coverage

- **DataLoader**: Data loading, preprocessing, validation
- **TrendFollowingStrategy**: Position calculation, backtesting, metrics
- **Utility Functions**: Helper functions, data manipulation
- **Configuration**: Config loading and validation

## Performance Metrics

The strategy calculates comprehensive performance metrics:

### Return Metrics
- Total Return
- Annualized Return
- Volatility (annualized)
- Sharpe Ratio

### Risk Metrics
- Maximum Drawdown
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Downside Deviation

### Distribution Metrics
- Skewness
- Kurtosis
- Win Rate
- Profit Factor

### Benchmark Comparison
- Excess Return
- Tracking Error
- Information Ratio
- Beta

## Visualization

The system generates several types of plots:

1. **Asset Availability**: Number of available assets over time
2. **Correlation Heatmap**: Asset correlation matrix for specific years
3. **Performance Dashboard**: Cumulative returns, drawdowns, rolling metrics
4. **Momentum Analysis**: Persistence analysis across lookback periods

## Configuration Options

### Environment Configurations

- **Default**: Standard settings for general use
- **Development**: Reduced plotting, verbose logging
- **Production**: High-quality plots, minimal logging

### Key Parameters

- **Lookback Periods**: 14-60 days (configurable)
- **Signal Lag**: 2 days (configurable)
- **Position Limits**: Maximum 100% per asset (configurable)
- **Risk-Free Rate**: 2% annual (configurable)

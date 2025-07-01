import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import logging
from tqdm import tqdm

from ..config import StrategyConfig, get_config
from ..analysis.trend_analyzer import TrendAnalyzer


class TrendFollowingStrategy:
    """
    Implementation of a cross-asset trend following strategy.
    
    This strategy uses multiple lookback periods to generate momentum signals
    and calculates normalized positions based on signal strength across assets.
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        Initialize the TrendFollowingStrategy.
        
        Args:
            config: Strategy configuration object. If None, uses default config.
        """
        self.config = config or get_config().strategy
        self.logger = logging.getLogger(__name__)
        self.trend_analyzer = TrendAnalyzer()
        
        # Strategy state
        self.signals: Optional[pd.DataFrame] = None
        self.positions: Optional[pd.DataFrame] = None
        self.strategy_returns: Optional[pd.DataFrame] = None
        
    def compute_positions_trend(self, 
                              returns: pd.DataFrame, 
                              periods_long: Optional[int] = None) -> pd.DataFrame:
        """
        Compute trend following positions based on multiple lookback periods.
        
        This is the core position calculation function that averages signals
        across multiple lookback periods and normalizes positions.
        
        Args:
            returns: DataFrame with return data.
            periods_long: Number of lookback periods to use. If None, uses config range.
            
        Returns:
            DataFrame with normalized positions (-1 to 1).
        """
        if periods_long is None:
            periods_long = self.config.max_lookback_days - self.config.min_lookback_days + 1
        
        # Initialize signal accumulator
        signal = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
        
        # Calculate signals for each lookback period
        start_period = self.config.min_lookback_days
        for period in range(start_period, start_period + periods_long):
            # Calculate cumulative returns over the period
            cumulative_returns = returns.rolling(window=period).sum()
            
            # Generate signal based on sign of cumulative returns
            return_sign = np.sign(cumulative_returns)
            
            # Add to signal accumulator
            signal = signal + return_sign.fillna(0)
        
        # Calculate total allocated signal per day (sum of absolute values)
        allocated = np.abs(signal).sum(axis=1)
        
        # Avoid division by zero
        allocated[allocated == 0] = np.nan
        
        # Normalize positions by total allocation
        position = signal.div(allocated, axis=0)
        
        # Apply position size constraints
        position = position.clip(
            lower=-self.config.max_position_size,
            upper=self.config.max_position_size
        )
        
        # Set minimum position threshold
        position = position.where(
            np.abs(position) >= self.config.min_position_size, 
            0
        )
        
        self.positions = position
        self.logger.info(f"Computed positions: {position.shape}")
        
        return position
    
    def generate_signals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using the trend following approach.
        
        Args:
            returns: DataFrame with return data.
            
        Returns:
            DataFrame with trading signals.
        """
        # Use the multi-period momentum calculation from TrendAnalyzer
        signals = self.trend_analyzer.calculate_multi_period_momentum(
            returns,
            self.config.min_lookback_days,
            self.config.max_lookback_days
        )
        
        self.signals = signals
        self.logger.info(f"Generated signals: {signals.shape}")
        
        return signals
    
    def calculate_strategy_returns(self, 
                                 returns: pd.DataFrame,
                                 positions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate strategy returns based on positions and market returns.
        
        Args:
            returns: DataFrame with market returns.
            positions: DataFrame with positions. If None, uses computed positions.
            
        Returns:
            DataFrame with strategy returns.
        """
        if positions is None:
            if self.positions is None:
                raise ValueError("No positions available. Call compute_positions_trend() first.")
            positions = self.positions
        
        # Apply signal lag for realistic trading
        lagged_positions = positions.shift(self.config.signal_lag_days)
        
        # Calculate strategy returns
        strategy_returns = returns * lagged_positions
        
        self.strategy_returns = strategy_returns
        self.logger.info(f"Calculated strategy returns: {strategy_returns.shape}")
        
        return strategy_returns
    
    def backtest(self, 
                returns: pd.DataFrame,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None) -> Dict:
        """
        Run complete backtest of the trend following strategy.
        
        Args:
            returns: DataFrame with return data.
            start_date: Start date for backtest. If None, uses all data.
            end_date: End date for backtest. If None, uses all data.
            
        Returns:
            Dictionary with backtest results.
        """
        self.logger.info("Starting trend following strategy backtest...")
        
        # Filter data by date range if specified
        if start_date or end_date:
            returns = returns.loc[start_date:end_date]
        
        # Step 1: Compute positions
        positions = self.compute_positions_trend(returns)
        
        # Step 2: Calculate strategy returns
        strategy_returns = self.calculate_strategy_returns(returns, positions)
        
        # Step 3: Calculate portfolio returns (equal weight across assets)
        portfolio_returns = strategy_returns.mean(axis=1, skipna=True)
        
        # Step 4: Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            portfolio_returns, 
            returns.mean(axis=1, skipna=True)  # Benchmark: equal weight buy-and-hold
        )
        
        # Step 5: Compile results
        backtest_results = {
            "portfolio_returns": portfolio_returns,
            "strategy_returns": strategy_returns,
            "positions": positions,
            "performance_metrics": performance_metrics,
            "backtest_period": {
                "start": returns.index.min().strftime("%Y-%m-%d"),
                "end": returns.index.max().strftime("%Y-%m-%d"),
                "total_days": len(returns)
            }
        }
        
        self.logger.info("Backtest completed successfully")
        
        return backtest_results
    
    def _calculate_performance_metrics(self, 
                                     strategy_returns: pd.Series,
                                     benchmark_returns: pd.Series) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            strategy_returns: Series with strategy returns.
            benchmark_returns: Series with benchmark returns.
            
        Returns:
            Dictionary with performance metrics.
        """
        # Remove NaN values
        strategy_returns = strategy_returns.dropna()
        benchmark_returns = benchmark_returns.dropna()
        
        # Align series
        common_index = strategy_returns.index.intersection(benchmark_returns.index)
        strategy_returns = strategy_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        
        # Calculate metrics
        metrics = {
            "total_return": strategy_returns.sum(),
            "annualized_return": strategy_returns.mean() * 252,
            "volatility": strategy_returns.std() * np.sqrt(252),
            "sharpe_ratio": self._calculate_sharpe_ratio(strategy_returns),
            "max_drawdown": self._calculate_max_drawdown(strategy_returns),
            "win_rate": (strategy_returns > 0).mean(),
            "benchmark_total_return": benchmark_returns.sum(),
            "benchmark_annualized_return": benchmark_returns.mean() * 252,
            "benchmark_volatility": benchmark_returns.std() * np.sqrt(252),
            "benchmark_sharpe_ratio": self._calculate_sharpe_ratio(benchmark_returns),
            "excess_return": strategy_returns.sum() - benchmark_returns.sum(),
            "information_ratio": self._calculate_information_ratio(strategy_returns, benchmark_returns)
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (0.02 / 252)  # Assuming 2% risk-free rate
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()
    
    def _calculate_information_ratio(self, 
                                   strategy_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> float:
        """Calculate information ratio."""
        excess_returns = strategy_returns - benchmark_returns
        if excess_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def get_position_statistics(self) -> Dict:
        """
        Get statistics about position allocation.
        
        Returns:
            Dictionary with position statistics.
        """
        if self.positions is None:
            return {"error": "No positions calculated"}
        
        positions = self.positions
        
        stats = {
            "average_positions_per_day": positions.notna().sum(axis=1).mean(),
            "max_positions_per_day": positions.notna().sum(axis=1).max(),
            "min_positions_per_day": positions.notna().sum(axis=1).min(),
            "average_position_size": np.abs(positions).mean().mean(),
            "long_position_ratio": (positions > 0).sum().sum() / positions.notna().sum().sum(),
            "short_position_ratio": (positions < 0).sum().sum() / positions.notna().sum().sum(),
            "zero_position_ratio": (positions == 0).sum().sum() / positions.notna().sum().sum()
        }
        
        return stats

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.strategy.trend_following import TrendFollowingStrategy
from src.config import StrategyConfig


class TestTrendFollowingStrategy:
    """Test cases for TrendFollowingStrategy class."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data for testing."""
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        
        # Create returns with some trend patterns
        returns = pd.DataFrame({
            "asset1": np.random.normal(0.001, 0.02, len(dates)),
            "asset2": np.random.normal(-0.0005, 0.015, len(dates)),
            "asset3": np.random.normal(0.0008, 0.025, len(dates))
        }, index=dates)
        
        # Add some trend patterns
        returns.iloc[50:100, 0] += 0.005  # Positive trend for asset1
        returns.iloc[150:200, 1] -= 0.003  # Negative trend for asset2
        
        return returns
    
    @pytest.fixture
    def strategy_config(self):
        """Create test strategy configuration."""
        return StrategyConfig(
            min_lookback_days=5,
            max_lookback_days=20,
            signal_lag_days=1,
            max_position_size=0.5
        )
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = TrendFollowingStrategy()
        assert strategy.config is not None
        assert strategy.signals is None
        assert strategy.positions is None
        assert strategy.strategy_returns is None
    
    def test_init_with_config(self, strategy_config):
        """Test strategy initialization with custom config."""
        strategy = TrendFollowingStrategy(strategy_config)
        assert strategy.config.min_lookback_days == 5
        assert strategy.config.max_lookback_days == 20
    
    def test_compute_positions_trend(self, sample_returns, strategy_config):
        """Test position computation."""
        strategy = TrendFollowingStrategy(strategy_config)
        positions = strategy.compute_positions_trend(sample_returns)
        
        assert isinstance(positions, pd.DataFrame)
        assert positions.shape == sample_returns.shape
        assert strategy.positions is not None
        
        # Check position constraints
        assert positions.abs().max().max() <= strategy_config.max_position_size
        
        # Check that positions sum to approximately 1 or 0 (normalized)
        position_sums = positions.abs().sum(axis=1).dropna()
        assert all(pos_sum <= 1.01 for pos_sum in position_sums)  # Allow small numerical errors
    
    def test_generate_signals(self, sample_returns, strategy_config):
        """Test signal generation."""
        strategy = TrendFollowingStrategy(strategy_config)
        signals = strategy.generate_signals(sample_returns)
        
        assert isinstance(signals, pd.DataFrame)
        assert signals.shape == sample_returns.shape
        assert strategy.signals is not None
        
        # Signals should be between -1 and 1
        assert signals.min().min() >= -1
        assert signals.max().max() <= 1
    
    def test_calculate_strategy_returns(self, sample_returns, strategy_config):
        """Test strategy returns calculation."""
        strategy = TrendFollowingStrategy(strategy_config)
        
        # First compute positions
        positions = strategy.compute_positions_trend(sample_returns)
        
        # Then calculate strategy returns
        strategy_returns = strategy.calculate_strategy_returns(sample_returns, positions)
        
        assert isinstance(strategy_returns, pd.DataFrame)
        assert strategy_returns.shape == sample_returns.shape
        assert strategy.strategy_returns is not None
        
        # Check that returns are reasonable (not too extreme)
        assert strategy_returns.abs().max().max() < 1.0  # No single day return > 100%
    
    def test_backtest(self, sample_returns, strategy_config):
        """Test complete backtest."""
        strategy = TrendFollowingStrategy(strategy_config)
        results = strategy.backtest(sample_returns)
        
        assert isinstance(results, dict)
        assert "portfolio_returns" in results
        assert "strategy_returns" in results
        assert "positions" in results
        assert "performance_metrics" in results
        assert "backtest_period" in results
        
        # Check performance metrics
        metrics = results["performance_metrics"]
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "volatility" in metrics
        
        # Check that metrics are reasonable
        assert isinstance(metrics["total_return"], (int, float))
        assert isinstance(metrics["sharpe_ratio"], (int, float))
        assert metrics["max_drawdown"] <= 0  # Drawdown should be negative
    
    def test_backtest_with_date_range(self, sample_returns, strategy_config):
        """Test backtest with specific date range."""
        strategy = TrendFollowingStrategy(strategy_config)
        
        start_date = "2020-06-01"
        end_date = "2020-08-31"
        
        results = strategy.backtest(sample_returns, start_date, end_date)
        
        # Check that backtest period is correct
        period_info = results["backtest_period"]
        assert period_info["start"] >= start_date
        assert period_info["end"] <= end_date
    
    def test_get_position_statistics(self, sample_returns, strategy_config):
        """Test position statistics calculation."""
        strategy = TrendFollowingStrategy(strategy_config)
        
        # First run backtest to generate positions
        strategy.backtest(sample_returns)
        
        # Get position statistics
        stats = strategy.get_position_statistics()
        
        assert isinstance(stats, dict)
        assert "average_positions_per_day" in stats
        assert "max_positions_per_day" in stats
        assert "long_position_ratio" in stats
        assert "short_position_ratio" in stats
        
        # Check that ratios sum to approximately 1
        long_ratio = stats["long_position_ratio"]
        short_ratio = stats["short_position_ratio"]
        zero_ratio = stats.get("zero_position_ratio", 0)
        
        assert 0 <= long_ratio <= 1
        assert 0 <= short_ratio <= 1
        assert abs(long_ratio + short_ratio + zero_ratio - 1) < 0.1  # Allow for rounding
    
    def test_performance_metrics_calculation(self, sample_returns, strategy_config):
        """Test individual performance metrics calculations."""
        strategy = TrendFollowingStrategy(strategy_config)
        
        # Create simple test returns
        test_returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        benchmark_returns = pd.Series([0.005, 0.002, 0.01, -0.008, 0.012])
        
        metrics = strategy._calculate_performance_metrics(test_returns, benchmark_returns)
        
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "excess_return" in metrics
        
        # Check that excess return is calculated correctly
        expected_excess = test_returns.sum() - benchmark_returns.sum()
        assert abs(metrics["excess_return"] - expected_excess) < 1e-10
    
    def test_sharpe_ratio_calculation(self, strategy_config):
        """Test Sharpe ratio calculation."""
        strategy = TrendFollowingStrategy(strategy_config)
        
        # Test with positive returns
        positive_returns = pd.Series([0.01, 0.02, 0.015, 0.008, 0.012])
        sharpe = strategy._calculate_sharpe_ratio(positive_returns)
        assert sharpe > 0
        
        # Test with zero volatility
        zero_vol_returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
        sharpe_zero_vol = strategy._calculate_sharpe_ratio(zero_vol_returns)
        assert sharpe_zero_vol == 0
    
    def test_max_drawdown_calculation(self, strategy_config):
        """Test maximum drawdown calculation."""
        strategy = TrendFollowingStrategy(strategy_config)
        
        # Test with known drawdown pattern
        returns = pd.Series([0.1, -0.05, -0.03, 0.02, -0.08, 0.15])
        max_dd = strategy._calculate_max_drawdown(returns)
        
        assert max_dd <= 0  # Drawdown should be negative
        assert isinstance(max_dd, (int, float))
    
    def test_empty_returns_handling(self, strategy_config):
        """Test handling of empty returns data."""
        strategy = TrendFollowingStrategy(strategy_config)

        empty_returns = pd.DataFrame()

        # Empty DataFrame should return empty positions
        positions = strategy.compute_positions_trend(empty_returns)
        assert positions.empty

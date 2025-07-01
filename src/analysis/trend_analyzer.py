import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
import logging
from scipy.stats import pearsonr
from tqdm import tqdm

from ..config import StrategyConfig, AnalysisConfig, get_config


class TrendAnalyzer:
    """
    Analyzes trend and momentum patterns in asset return data.
    
    This class provides methods for calculating momentum signals,
    analyzing time-series momentum effects, and researching optimal
    lookback periods for trend following strategies.
    """
    
    def __init__(self, 
                 strategy_config: Optional[StrategyConfig] = None,
                 analysis_config: Optional[AnalysisConfig] = None):
        """
        Initialize the TrendAnalyzer.
        
        Args:
            strategy_config: Strategy configuration object.
            analysis_config: Analysis configuration object.
        """
        config = get_config()
        self.strategy_config = strategy_config or config.strategy
        self.analysis_config = analysis_config or config.analysis
        self.logger = logging.getLogger(__name__)
        
    def calculate_momentum_signal(self, 
                                returns: pd.DataFrame, 
                                lookback_period: int) -> pd.DataFrame:
        """
        Calculate momentum signal for a given lookback period.
        
        Args:
            returns: DataFrame with return data.
            lookback_period: Number of days to look back for momentum calculation.
            
        Returns:
            DataFrame with momentum signals (-1, 0, 1).
        """
        # Calculate cumulative returns over lookback period
        cumulative_returns = returns.rolling(window=lookback_period).sum()
        
        # Generate signals based on sign of cumulative returns
        signals = np.sign(cumulative_returns)
        
        return signals
    
    def calculate_multi_period_momentum(self, 
                                      returns: pd.DataFrame,
                                      min_period: Optional[int] = None,
                                      max_period: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate momentum signals across multiple lookback periods.
        
        Args:
            returns: DataFrame with return data.
            min_period: Minimum lookback period. If None, uses config.
            max_period: Maximum lookback period. If None, uses config.
            
        Returns:
            DataFrame with averaged momentum signals.
        """
        min_period = min_period or self.strategy_config.min_lookback_days
        max_period = max_period or self.strategy_config.max_lookback_days
        
        # Initialize signal accumulator
        signal_sum = pd.DataFrame(0, index=returns.index, columns=returns.columns)
        
        # Calculate signals for each lookback period
        for period in range(min_period, max_period + 1):
            period_signal = self.calculate_momentum_signal(returns, period)
            signal_sum += period_signal.fillna(0)
        
        # Average the signals
        num_periods = max_period - min_period + 1
        averaged_signals = signal_sum / num_periods
        
        return averaged_signals
    
    def analyze_momentum_persistence(self, 
                                   returns: pd.DataFrame,
                                   lookback_periods: Optional[List[int]] = None) -> Dict:
        """
        Analyze momentum persistence across different lookback periods.
        
        Args:
            returns: DataFrame with return data.
            lookback_periods: List of lookback periods to analyze.
            
        Returns:
            Dictionary with momentum persistence analysis.
        """
        if lookback_periods is None:
            lookback_periods = list(range(
                self.strategy_config.min_lookback_days,
                self.strategy_config.max_lookback_days + 1,
                5  # Step of 5 days for efficiency
            ))
        
        persistence_results = {}
        
        for period in tqdm(lookback_periods, desc="Analyzing momentum persistence"):
            # Calculate momentum signals
            signals = self.calculate_momentum_signal(returns, period)
            
            # Calculate persistence (correlation between consecutive signals)
            persistence_scores = []
            for asset in returns.columns:
                asset_signals = signals[asset].dropna()
                if len(asset_signals) > 1:
                    # Calculate correlation between t and t-1 signals
                    correlation, _ = pearsonr(
                        asset_signals[1:].values,
                        asset_signals[:-1].values
                    )
                    persistence_scores.append(correlation)
            
            persistence_results[period] = {
                "mean_persistence": np.mean(persistence_scores),
                "median_persistence": np.median(persistence_scores),
                "std_persistence": np.std(persistence_scores),
                "positive_persistence_ratio": np.sum(np.array(persistence_scores) > 0) / len(persistence_scores)
            }
        
        return persistence_results
    
    def calculate_momentum_returns(self, 
                                 returns: pd.DataFrame,
                                 signals: pd.DataFrame,
                                 lag_days: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate returns from momentum strategy.
        
        Args:
            returns: DataFrame with return data.
            signals: DataFrame with momentum signals.
            lag_days: Number of days to lag signals. If None, uses config.
            
        Returns:
            DataFrame with strategy returns.
        """
        lag_days = lag_days or self.strategy_config.signal_lag_days
        
        # Lag the signals to simulate realistic trading
        lagged_signals = signals.shift(lag_days)
        
        # Calculate strategy returns
        strategy_returns = returns * lagged_signals
        
        return strategy_returns
    
    def optimize_lookback_period(self, 
                               returns: pd.DataFrame,
                               lookback_range: Optional[Tuple[int, int]] = None,
                               metric: str = "sharpe_ratio") -> Dict:
        """
        Optimize the lookback period based on specified metric.
        
        Args:
            returns: DataFrame with return data.
            lookback_range: Tuple of (min, max) lookback periods.
            metric: Optimization metric ("sharpe_ratio", "total_return", "volatility").
            
        Returns:
            Dictionary with optimization results.
        """
        if lookback_range is None:
            lookback_range = (
                self.strategy_config.min_lookback_days,
                self.strategy_config.max_lookback_days
            )
        
        min_period, max_period = lookback_range
        optimization_results = {}
        
        for period in tqdm(range(min_period, max_period + 1), desc="Optimizing lookback period"):
            # Calculate signals and returns for this period
            signals = self.calculate_momentum_signal(returns, period)
            strategy_returns = self.calculate_momentum_returns(returns, signals)
            
            # Calculate portfolio returns (equal weight)
            portfolio_returns = strategy_returns.mean(axis=1, skipna=True)
            
            # Calculate metrics
            if metric == "sharpe_ratio":
                metric_value = self._calculate_sharpe_ratio(portfolio_returns)
            elif metric == "total_return":
                metric_value = portfolio_returns.sum()
            elif metric == "volatility":
                metric_value = -portfolio_returns.std()  # Negative for minimization
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            optimization_results[period] = {
                "metric_value": metric_value,
                "total_return": portfolio_returns.sum(),
                "volatility": portfolio_returns.std(),
                "sharpe_ratio": self._calculate_sharpe_ratio(portfolio_returns)
            }
        
        # Find optimal period
        optimal_period = max(optimization_results.keys(), 
                           key=lambda x: optimization_results[x]["metric_value"])
        
        return {
            "optimal_period": optimal_period,
            "optimal_metric_value": optimization_results[optimal_period]["metric_value"],
            "all_results": optimization_results
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio for a return series.
        
        Args:
            returns: Series with return data.
            
        Returns:
            Sharpe ratio.
        """
        if returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (self.analysis_config.risk_free_rate / 
                                  self.analysis_config.trading_days_per_year)
        
        return (excess_returns.mean() / returns.std()) * np.sqrt(self.analysis_config.trading_days_per_year)
    
    def plot_momentum_analysis(self, 
                             returns: pd.DataFrame,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive momentum analysis plots.
        
        Args:
            returns: DataFrame with return data.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure object.
        """
        # Analyze momentum persistence
        persistence_results = self.analyze_momentum_persistence(returns)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, 
                                                     figsize=(15, 10), 
                                                     dpi=self.analysis_config.dpi)
        
        # Plot 1: Momentum persistence by lookback period
        periods = list(persistence_results.keys())
        mean_persistence = [persistence_results[p]["mean_persistence"] for p in periods]
        
        ax1.plot(periods, mean_persistence, 'b-', linewidth=2, marker='o')
        ax1.set_title("Momentum Persistence by Lookback Period")
        ax1.set_xlabel("Lookback Period (days)")
        ax1.set_ylabel("Mean Persistence")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of persistence scores
        all_persistence = [persistence_results[p]["mean_persistence"] for p in periods]
        ax2.hist(all_persistence, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title("Distribution of Momentum Persistence")
        ax2.set_xlabel("Persistence Score")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Positive persistence ratio
        pos_persistence = [persistence_results[p]["positive_persistence_ratio"] for p in periods]
        ax3.plot(periods, pos_persistence, 'g-', linewidth=2, marker='s')
        ax3.set_title("Positive Persistence Ratio by Lookback Period")
        ax3.set_xlabel("Lookback Period (days)")
        ax3.set_ylabel("Positive Persistence Ratio")
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Persistence standard deviation
        std_persistence = [persistence_results[p]["std_persistence"] for p in periods]
        ax4.plot(periods, std_persistence, 'r-', linewidth=2, marker='^')
        ax4.set_title("Persistence Standard Deviation by Lookback Period")
        ax4.set_xlabel("Lookback Period (days)")
        ax4.set_ylabel("Standard Deviation")
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if self.analysis_config.save_plots and save_path:
            fig.savefig(save_path, dpi=self.analysis_config.dpi, bbox_inches='tight')
            self.logger.info(f"Saved momentum analysis plot to {save_path}")
        
        return fig

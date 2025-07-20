import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path

from ..config import AnalysisConfig, get_config


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies.
    
    This class provides methods for calculating performance metrics,
    risk analysis, and generating performance visualizations.
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the PerformanceAnalyzer.
        
        Args:
            config: Analysis configuration object. If None, uses default config.
        """
        self.config = config or get_config().analysis
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib style
        plt.style.use(self.config.style)
        
    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """
        Calculate cumulative returns from a return series.
        
        Args:
            returns: Series with period returns.
            
        Returns:
            Series with cumulative returns.
        """
        return (1 + returns.fillna(0)).cumprod() - 1
    
    def calculate_rolling_sharpe(self, 
                               returns: pd.Series, 
                               window: int = 252) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.
        
        Args:
            returns: Series with period returns.
            window: Rolling window size in days.
            
        Returns:
            Series with rolling Sharpe ratios.
        """
        excess_returns = returns - (self.config.risk_free_rate / 
                                  self.config.trading_days_per_year)
        
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(self.config.trading_days_per_year)
        
        return rolling_sharpe
    
    def calculate_rolling_volatility(self, 
                                   returns: pd.Series, 
                                   window: int = 252) -> pd.Series:
        """
        Calculate rolling volatility (annualized).
        
        Args:
            returns: Series with period returns.
            window: Rolling window size in days.
            
        Returns:
            Series with rolling volatility.
        """
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(self.config.trading_days_per_year)
        return rolling_vol
    
    def calculate_drawdown_series(self, returns: pd.Series) -> pd.Series:
        """
        Calculate drawdown series.
        
        Args:
            returns: Series with period returns.
            
        Returns:
            Series with drawdown values.
        """
        cumulative_returns = self.calculate_cumulative_returns(returns)
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / (1 + rolling_max)
        
        return drawdown
    
    def calculate_var(self, 
                     returns: pd.Series, 
                     confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series with period returns.
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR).
            
        Returns:
            VaR value.
        """
        return np.percentile(returns.dropna(), confidence_level * 100)
    
    def calculate_cvar(self, 
                      returns: pd.Series, 
                      confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: Series with period returns.
            confidence_level: Confidence level (e.g., 0.05 for 95% CVaR).
            
        Returns:
            CVaR value.
        """
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def generate_performance_report(self, 
                                  strategy_returns: pd.Series,
                                  benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Generate comprehensive performance report.
        
        Args:
            strategy_returns: Series with strategy returns.
            benchmark_returns: Series with benchmark returns (optional).
            
        Returns:
            Dictionary with performance metrics.
        """
        # Clean data
        strategy_returns = strategy_returns.dropna()
        
        # Basic metrics
        total_return = self.calculate_cumulative_returns(strategy_returns).iloc[-1]
        annualized_return = strategy_returns.mean() * self.config.trading_days_per_year
        volatility = strategy_returns.std() * np.sqrt(self.config.trading_days_per_year)
        
        # Risk metrics
        sharpe_ratio = ((strategy_returns.mean() - self.config.risk_free_rate / 
                        self.config.trading_days_per_year) / strategy_returns.std() * 
                       np.sqrt(self.config.trading_days_per_year))
        
        max_drawdown = self.calculate_drawdown_series(strategy_returns).min()
        var_95 = self.calculate_var(strategy_returns, 0.05)
        cvar_95 = self.calculate_cvar(strategy_returns, 0.05)
        
        # Additional metrics
        win_rate = (strategy_returns > 0).mean()
        avg_win = strategy_returns[strategy_returns > 0].mean()
        avg_loss = strategy_returns[strategy_returns < 0].mean()
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        # Skewness and kurtosis
        skewness = strategy_returns.skew()
        kurtosis = strategy_returns.kurtosis()
        
        report = {
            "return_metrics": {
                "total_return": total_return,
                "annualized_return": annualized_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio
            },
            "risk_metrics": {
                "max_drawdown": max_drawdown,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "downside_deviation": strategy_returns[strategy_returns < 0].std() * np.sqrt(252)
            },
            "distribution_metrics": {
                "skewness": skewness,
                "kurtosis": kurtosis,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_win": avg_win,
                "avg_loss": avg_loss
            },
            "period_info": {
                "start_date": strategy_returns.index.min().strftime("%Y-%m-%d"),
                "end_date": strategy_returns.index.max().strftime("%Y-%m-%d"),
                "total_days": len(strategy_returns),
                "trading_days": (strategy_returns != 0).sum()
            }
        }
        
        # Add benchmark comparison if provided
        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.dropna()
            common_index = strategy_returns.index.intersection(benchmark_returns.index)
            
            if len(common_index) > 0:
                strategy_aligned = strategy_returns.loc[common_index]
                benchmark_aligned = benchmark_returns.loc[common_index]
                
                excess_returns = strategy_aligned - benchmark_aligned
                tracking_error = excess_returns.std() * np.sqrt(self.config.trading_days_per_year)
                information_ratio = (excess_returns.mean() / excess_returns.std() * 
                                   np.sqrt(self.config.trading_days_per_year))
                
                benchmark_total_return = self.calculate_cumulative_returns(benchmark_aligned).iloc[-1]
                benchmark_volatility = benchmark_aligned.std() * np.sqrt(self.config.trading_days_per_year)
                
                report["benchmark_comparison"] = {
                    "benchmark_total_return": benchmark_total_return,
                    "benchmark_volatility": benchmark_volatility,
                    "excess_return": total_return - benchmark_total_return,
                    "tracking_error": tracking_error,
                    "information_ratio": information_ratio,
                    "beta": np.cov(strategy_aligned, benchmark_aligned)[0, 1] / np.var(benchmark_aligned)
                }
        
        return report
    
    def plot_performance_dashboard(self, 
                                 strategy_returns: pd.Series,
                                 benchmark_returns: Optional[pd.Series] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive performance dashboard.
        
        Args:
            strategy_returns: Series with strategy returns.
            benchmark_returns: Series with benchmark returns (optional).
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure object.
        """
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, 
                                                     figsize=(16, 12), 
                                                     dpi=self.config.dpi)
        
        # Plot 1: Cumulative returns
        cumulative_strategy = self.calculate_cumulative_returns(strategy_returns)
        ax1.plot(cumulative_strategy.index, cumulative_strategy.values, 
                label="Strategy", linewidth=2, color="blue")
        
        if benchmark_returns is not None:
            cumulative_benchmark = self.calculate_cumulative_returns(benchmark_returns)
            ax1.plot(cumulative_benchmark.index, cumulative_benchmark.values, 
                    label="Benchmark", linewidth=2, color="red", alpha=0.7)
        
        ax1.set_title("Cumulative Returns", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Cumulative Return")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        drawdown = self.calculate_drawdown_series(strategy_returns)
        ax2.fill_between(drawdown.index, drawdown.values, 0, 
                        alpha=0.3, color="red", label="Drawdown")
        ax2.plot(drawdown.index, drawdown.values, color="red", linewidth=1)
        ax2.set_title("Drawdown", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Drawdown")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Rolling Sharpe ratio
        rolling_sharpe = self.calculate_rolling_sharpe(strategy_returns)
        ax3.plot(rolling_sharpe.index, rolling_sharpe.values, 
                color="green", linewidth=2)
        ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax3.set_title("Rolling Sharpe Ratio (252-day)", fontsize=14, fontweight='bold')
        ax3.set_ylabel("Sharpe Ratio")
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Return distribution
        ax4.hist(strategy_returns.dropna(), bins=50, alpha=0.7, 
                color="skyblue", edgecolor="black", density=True)
        ax4.axvline(strategy_returns.mean(), color="red", linestyle="--", 
                   label=f"Mean: {strategy_returns.mean():.4f}")
        ax4.set_title("Return Distribution", fontsize=14, fontweight='bold')
        ax4.set_xlabel("Daily Return")
        ax4.set_ylabel("Density")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if self.config.save_plots and save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            self.logger.info(f"Saved performance dashboard to {save_path}")
        
        return fig

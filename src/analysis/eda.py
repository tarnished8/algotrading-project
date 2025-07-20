import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, List
import logging
from pathlib import Path

from ..config import AnalysisConfig, get_config


class ExploratoryDataAnalysis:
    """
    Comprehensive Exploratory Data Analysis for asset price data.
    
    This class provides methods for analyzing correlations, visualizing data
    availability, and generating statistical insights about the asset universe.
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the EDA analyzer.
        
        Args:
            config: Analysis configuration object. If None, uses default config.
        """
        self.config = config or get_config().analysis
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib style
        plt.style.use(self.config.style)
        
    def plot_asset_availability(self, 
                               data: pd.DataFrame, 
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the number of available assets over time.
        
        Args:
            data: DataFrame with asset price data.
            save_path: Path to save the plot. If None, uses config settings.
            
        Returns:
            Matplotlib figure object.
        """
        # Calculate available assets per day
        available_assets = data.notnull().sum(axis=1)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        ax.plot(available_assets.index, available_assets.values, 
                label="Available Assets", color="blue", linewidth=1.5)
        
        ax.set_title("Number of Available Assets Over Time", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Number of Assets", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add summary statistics as text
        stats_text = f"Mean: {available_assets.mean():.1f}\n"
        stats_text += f"Max: {available_assets.max()}\n"
        stats_text += f"Min: {available_assets.min()}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save if requested
        if self.config.save_plots:
            save_path = save_path or f"{self.config.output_dir}/asset_availability.{self.config.plot_format}"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            self.logger.info(f"Saved asset availability plot to {save_path}")
        
        return fig
    
    def calculate_correlation_matrix(self, 
                                   returns: pd.DataFrame, 
                                   year: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix for a specific year or entire dataset.
        
        Args:
            returns: DataFrame with return data.
            year: Specific year to analyze. If None, uses config year.
            
        Returns:
            Correlation matrix DataFrame.
        """
        year = year or self.config.correlation_year
        
        # Filter data for the specified year
        if year:
            year_data = returns[returns.index.year == year]
            if year_data.empty:
                self.logger.warning(f"No data available for year {year}")
                year_data = returns
        else:
            year_data = returns
        
        # Calculate correlation matrix
        correlation_matrix = year_data.corr()
        
        self.logger.info(f"Calculated correlation matrix for year {year}: {correlation_matrix.shape}")
        
        return correlation_matrix
    
    def plot_correlation_heatmap(self, 
                               returns: pd.DataFrame, 
                               year: Optional[int] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a correlation heatmap for the specified year.
        
        Args:
            returns: DataFrame with return data.
            year: Specific year to analyze. If None, uses config year.
            save_path: Path to save the plot. If None, uses config settings.
            
        Returns:
            Matplotlib figure object.
        """
        year = year or self.config.correlation_year
        correlation_matrix = self.calculate_correlation_matrix(returns, year)
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Create heatmap with custom colormap
        sns.heatmap(correlation_matrix, 
                   annot=False, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   ax=ax,
                   cbar_kws={"shrink": 0.8})
        
        ax.set_title(f"Asset Correlation Matrix - {year}", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if requested
        if self.config.save_plots:
            save_path = save_path or f"{self.config.output_dir}/correlation_heatmap_{year}.{self.config.plot_format}"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            self.logger.info(f"Saved correlation heatmap to {save_path}")
        
        return fig
    
    def find_extreme_correlations(self, 
                                returns: pd.DataFrame, 
                                year: Optional[int] = None) -> Dict[str, Tuple[str, str, float]]:
        """
        Find asset pairs with highest and lowest correlations.
        
        Args:
            returns: DataFrame with return data.
            year: Specific year to analyze. If None, uses config year.
            
        Returns:
            Dictionary with highest and lowest correlation pairs.
        """
        correlation_matrix = self.calculate_correlation_matrix(returns, year)
        
        # Get upper triangle of correlation matrix (excluding diagonal)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        upper_triangle = correlation_matrix.where(mask)
        
        # Find highest correlation
        max_corr_idx = upper_triangle.stack().idxmax()
        max_corr_value = upper_triangle.stack().max()
        
        # Find lowest correlation
        min_corr_idx = upper_triangle.stack().idxmin()
        min_corr_value = upper_triangle.stack().min()
        
        results = {
            "highest": (max_corr_idx[0], max_corr_idx[1], max_corr_value),
            "lowest": (min_corr_idx[0], min_corr_idx[1], min_corr_value)
        }
        
        self.logger.info(f"Highest correlation: {max_corr_idx[0]} - {max_corr_idx[1]}: {max_corr_value:.4f}")
        self.logger.info(f"Lowest correlation: {min_corr_idx[0]} - {min_corr_idx[1]}: {min_corr_value:.4f}")
        
        return results
    
    def analyze_correlation_structure(self, returns: pd.DataFrame, year: Optional[int] = None) -> Dict:
        """
        Analyze the correlation structure to identify potential asset blocks.
        
        Args:
            returns: DataFrame with return data.
            year: Specific year to analyze. If None, uses config year.
            
        Returns:
            Dictionary with correlation structure analysis.
        """
        correlation_matrix = self.calculate_correlation_matrix(returns, year)
        
        # Calculate summary statistics
        corr_values = correlation_matrix.values
        upper_triangle = corr_values[np.triu_indices_from(corr_values, k=1)]
        
        analysis = {
            "mean_correlation": np.mean(upper_triangle),
            "median_correlation": np.median(upper_triangle),
            "std_correlation": np.std(upper_triangle),
            "min_correlation": np.min(upper_triangle),
            "max_correlation": np.max(upper_triangle),
            "positive_correlations": np.sum(upper_triangle > 0) / len(upper_triangle),
            "high_correlations": np.sum(upper_triangle > 0.5) / len(upper_triangle),
            "negative_correlations": np.sum(upper_triangle < 0) / len(upper_triangle)
        }
        
        return analysis
    
    def generate_summary_statistics(self, 
                                  data: pd.DataFrame, 
                                  returns: pd.DataFrame) -> Dict:
        """
        Generate comprehensive summary statistics for the dataset.
        
        Args:
            data: DataFrame with price data.
            returns: DataFrame with return data.
            
        Returns:
            Dictionary with summary statistics.
        """
        summary = {
            "data_overview": {
                "total_assets": data.shape[1],
                "total_observations": data.shape[0],
                "date_range": {
                    "start": data.index.min().strftime("%Y-%m-%d"),
                    "end": data.index.max().strftime("%Y-%m-%d"),
                    "total_days": (data.index.max() - data.index.min()).days
                }
            },
            "data_quality": {
                "missing_values_total": data.isnull().sum().sum(),
                "missing_percentage": (data.isnull().sum().sum() / data.size) * 100,
                "assets_with_complete_data": (data.isnull().sum() == 0).sum(),
                "avg_availability_per_asset": (1 - data.isnull().sum() / len(data)).mean()
            },
            "return_statistics": {
                "mean_daily_return": returns.mean().mean(),
                "median_daily_return": returns.median().median(),
                "std_daily_return": returns.std().mean(),
                "skewness": returns.skew().mean(),
                "kurtosis": returns.kurtosis().mean()
            }
        }
        
        return summary
    
    def create_comprehensive_report(self, 
                                  data: pd.DataFrame, 
                                  returns: pd.DataFrame,
                                  save_path: Optional[str] = None) -> Dict:
        """
        Create a comprehensive EDA report with all analyses.
        
        Args:
            data: DataFrame with price data.
            returns: DataFrame with return data.
            save_path: Path to save plots. If None, uses config settings.
            
        Returns:
            Dictionary with complete analysis results.
        """
        self.logger.info("Starting comprehensive EDA report generation...")
        
        # Generate all analyses
        report = {
            "summary_statistics": self.generate_summary_statistics(data, returns),
            "correlation_analysis": self.analyze_correlation_structure(returns),
            "extreme_correlations": self.find_extreme_correlations(returns)
        }
        
        # Generate plots
        self.plot_asset_availability(data, save_path)
        self.plot_correlation_heatmap(returns, save_path=save_path)
        
        self.logger.info("Comprehensive EDA report completed")
        
        return report

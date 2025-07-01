#!/usr/bin/env python3
import logging
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import get_config, get_config_by_env
from src.data.data_loader import DataLoader
from src.analysis.eda import ExploratoryDataAnalysis
from src.analysis.trend_analyzer import TrendAnalyzer
from src.strategy.trend_following import TrendFollowingStrategy
from src.analysis.performance_analyzer import PerformanceAnalyzer


def setup_logging(verbose: bool = True) -> None:
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging.
    """
    level = logging.INFO if verbose else logging.WARNING
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('algotrading_strategy.log')
        ]
    )


def run_eda_analysis(data: pd.DataFrame, returns: pd.DataFrame, config) -> None:
    """
    Run exploratory data analysis.
    
    Args:
        data: Price data DataFrame.
        returns: Returns data DataFrame.
        config: Configuration object.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Exploratory Data Analysis...")
    
    # Initialize EDA analyzer
    eda = ExploratoryDataAnalysis(config.analysis)
    
    # Generate comprehensive EDA report
    eda_report = eda.create_comprehensive_report(data, returns)
    
    # Print summary statistics
    print("EXPLORATORY DATA ANALYSIS SUMMARY")
    
    summary = eda_report["summary_statistics"]
    print(f"Dataset Overview:")
    print(f"  - Total Assets: {summary['data_overview']['total_assets']}")
    print(f"  - Total Observations: {summary['data_overview']['total_observations']}")
    print(f"  - Date Range: {summary['data_overview']['date_range']['start']} to {summary['data_overview']['date_range']['end']}")
    print(f"  - Missing Data: {summary['data_quality']['missing_percentage']:.2f}%")
    
    print(f"\nReturn Statistics:")
    print(f"  - Mean Daily Return: {summary['return_statistics']['mean_daily_return']:.6f}")
    print(f"  - Daily Volatility: {summary['return_statistics']['std_daily_return']:.6f}")
    print(f"  - Skewness: {summary['return_statistics']['skewness']:.3f}")
    print(f"  - Kurtosis: {summary['return_statistics']['kurtosis']:.3f}")
    
    logger.info("EDA analysis completed")


def run_trend_analysis(returns: pd.DataFrame, config) -> None:
    """
    Run trend and momentum analysis.
    
    Args:
        returns: Returns data DataFrame.
        config: Configuration object.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Trend Analysis...")
    
    # Initialize trend analyzer
    trend_analyzer = TrendAnalyzer(config.strategy, config.analysis)
    
    # Optimize lookback period
    optimization_results = trend_analyzer.optimize_lookback_period(returns)
    
    print("TREND ANALYSIS SUMMARY")
    print(f"Optimal Lookback Period: {optimization_results['optimal_period']} days")
    print(f"Optimal Sharpe Ratio: {optimization_results['optimal_metric_value']:.4f}")
    
    # Generate momentum analysis plots
    if config.analysis.save_plots:
        plot_path = f"{config.analysis.output_dir}/momentum_analysis.{config.analysis.plot_format}"
        trend_analyzer.plot_momentum_analysis(returns, plot_path)
    
    logger.info("Trend analysis completed")


def run_strategy_backtest(returns: pd.DataFrame, config) -> dict:
    """
    Run the complete strategy backtest.
    
    Args:
        returns: Returns data DataFrame.
        config: Configuration object.
        
    Returns:
        Dictionary with backtest results.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Strategy Backtest...")
    
    # Initialize strategy
    strategy = TrendFollowingStrategy(config.strategy)
    
    # Run backtest
    backtest_results = strategy.backtest(returns)
    
    # Print performance summary
    print("STRATEGY BACKTEST SUMMARY")
    
    metrics = backtest_results["performance_metrics"]
    print(f"Performance Metrics:")
    print(f"  - Total Return: {metrics['total_return']:.4f} ({metrics['total_return']*100:.2f}%)")
    print(f"  - Annualized Return: {metrics['annualized_return']:.4f} ({metrics['annualized_return']*100:.2f}%)")
    print(f"  - Volatility: {metrics['volatility']:.4f} ({metrics['volatility']*100:.2f}%)")
    print(f"  - Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"  - Maximum Drawdown: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
    print(f"  - Win Rate: {metrics['win_rate']:.4f} ({metrics['win_rate']*100:.2f}%)")
    
    if 'benchmark_total_return' in metrics:
        print(f"\nBenchmark Comparison:")
        print(f"  - Benchmark Return: {metrics['benchmark_total_return']:.4f} ({metrics['benchmark_total_return']*100:.2f}%)")
        print(f"  - Excess Return: {metrics['excess_return']:.4f} ({metrics['excess_return']*100:.2f}%)")
        print(f"  - Information Ratio: {metrics['information_ratio']:.4f}")
    
    # Get position statistics
    position_stats = strategy.get_position_statistics()
    print(f"\nPosition Statistics:")
    print(f"  - Average Positions per Day: {position_stats['average_positions_per_day']:.1f}")
    print(f"  - Long Position Ratio: {position_stats['long_position_ratio']:.2f}")
    print(f"  - Short Position Ratio: {position_stats['short_position_ratio']:.2f}")
    
    logger.info("Strategy backtest completed")
    
    return backtest_results


def run_performance_analysis(backtest_results: dict, config) -> None:
    """
    Run comprehensive performance analysis.
    
    Args:
        backtest_results: Results from strategy backtest.
        config: Configuration object.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Performance Analysis...")
    
    # Initialize performance analyzer
    perf_analyzer = PerformanceAnalyzer(config.analysis)
    
    # Extract returns
    portfolio_returns = backtest_results["portfolio_returns"]
    
    # Generate performance report
    performance_report = perf_analyzer.generate_performance_report(portfolio_returns)
    
    # Create performance dashboard
    if config.analysis.save_plots:
        dashboard_path = f"{config.analysis.output_dir}/performance_dashboard.{config.analysis.plot_format}"
        perf_analyzer.plot_performance_dashboard(portfolio_returns, save_path=dashboard_path)
    
    print("DETAILED PERFORMANCE ANALYSIS")
    
    # Print detailed metrics
    return_metrics = performance_report["return_metrics"]
    risk_metrics = performance_report["risk_metrics"]
    dist_metrics = performance_report["distribution_metrics"]
    
    print(f"Return Metrics:")
    print(f"  - Total Return: {return_metrics['total_return']:.4f}")
    print(f"  - Annualized Return: {return_metrics['annualized_return']:.4f}")
    print(f"  - Volatility: {return_metrics['volatility']:.4f}")
    print(f"  - Sharpe Ratio: {return_metrics['sharpe_ratio']:.4f}")
    
    print(f"\nRisk Metrics:")
    print(f"  - Maximum Drawdown: {risk_metrics['max_drawdown']:.4f}")
    print(f"  - VaR (95%): {risk_metrics['var_95']:.4f}")
    print(f"  - CVaR (95%): {risk_metrics['cvar_95']:.4f}")
    print(f"  - Downside Deviation: {risk_metrics['downside_deviation']:.4f}")
    
    print(f"\nDistribution Metrics:")
    print(f"  - Skewness: {dist_metrics['skewness']:.3f}")
    print(f"  - Kurtosis: {dist_metrics['kurtosis']:.3f}")
    print(f"  - Win Rate: {dist_metrics['win_rate']:.3f}")
    print(f"  - Profit Factor: {dist_metrics['profit_factor']:.3f}")
    
    logger.info("Performance analysis completed")


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Algorithmic Trading Strategy Analysis")
    parser.add_argument("--env", default="default", 
                       choices=["default", "development", "production"],
                       help="Configuration environment")
    parser.add_argument("--data-file", default=None,
                       help="Path to data file (overrides config)")
    parser.add_argument("--skip-eda", action="store_true",
                       help="Skip exploratory data analysis")
    parser.add_argument("--skip-trend", action="store_true",
                       help="Skip trend analysis")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config_by_env(args.env)
    
    # Override data file if specified
    if args.data_file:
        config.data.data_file = args.data_file
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        print("ALGORITHMIC TRADING STRATEGY ANALYSIS")
        print(f"Environment: {args.env}")
        print(f"Data File: {config.data.data_file}")
        print(f"Configuration: {config.strategy.min_lookback_days}-{config.strategy.max_lookback_days} day lookback")
        
        # Step 1: Load and preprocess data
        logger.info("Loading and preprocessing data...")
        data_loader = DataLoader(config.data)
        data, returns = data_loader.load_and_preprocess()
        
        # Validate data
        if not data_loader.validate_data(data):
            raise ValueError("Data validation failed")
        
        # Step 2: Exploratory Data Analysis (optional)
        if not args.skip_eda:
            run_eda_analysis(data, returns, config)
        
        # Step 3: Trend Analysis (optional)
        if not args.skip_trend:
            run_trend_analysis(returns, config)
        
        # Step 4: Strategy Backtest
        backtest_results = run_strategy_backtest(returns, config)
        
        # Step 5: Performance Analysis
        run_performance_analysis(backtest_results, config)
        
        print("ANALYSIS COMPLETED SUCCESSFULLY")

        if config.analysis.save_plots:
            print(f"Plots saved to: {config.analysis.output_dir}/")
        
        logger.info("All analyses completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        print(f"\nERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

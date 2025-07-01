#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import get_config
from src.data.data_loader import DataLoader
from src.analysis.eda import ExploratoryDataAnalysis
from src.analysis.trend_analyzer import TrendAnalyzer
from src.strategy.trend_following import TrendFollowingStrategy
from src.analysis.performance_analyzer import PerformanceAnalyzer


def main():
    """Run example analysis."""
    print("Algorithmic Trading Strategy - Example")
    
    # Get configuration
    config = get_config()
    print(f"Using configuration: {config.strategy.min_lookback_days}-{config.strategy.max_lookback_days} day lookback")
    
    try:
        # Step 1: Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        loader = DataLoader(config.data)
        data, returns = loader.load_and_preprocess()
        
        print(f"   Loaded data: {data.shape[0]} days, {data.shape[1]} assets")
        print(f"   Date range: {data.index.min().date()} to {data.index.max().date()}")
        
        # Step 2: Quick EDA
        print("\n2. Running exploratory data analysis...")
        eda = ExploratoryDataAnalysis(config.analysis)
        
        # Get data summary
        summary = eda.generate_summary_statistics(data, returns)
        print(f"   Missing data: {summary['data_quality']['missing_percentage']:.2f}%")
        print(f"   Mean daily return: {summary['return_statistics']['mean_daily_return']:.6f}")
        print(f"   Daily volatility: {summary['return_statistics']['std_daily_return']:.6f}")
        
        # Step 3: Trend analysis
        print("\n3. Analyzing trends and momentum...")
        trend_analyzer = TrendAnalyzer(config.strategy, config.analysis)
        
        # Optimize lookback period
        optimization = trend_analyzer.optimize_lookback_period(
            returns.iloc[:1000],  # Use subset for speed
            lookback_range=(14, 30)
        )
        print(f"   Optimal lookback period: {optimization['optimal_period']} days")
        print(f"   Optimal Sharpe ratio: {optimization['optimal_metric_value']:.4f}")
        
        # Step 4: Run strategy backtest
        print("\n4. Running strategy backtest...")
        strategy = TrendFollowingStrategy(config.strategy)
        
        # Run backtest on subset for demonstration
        subset_returns = returns.iloc[:1000]  # First 1000 days
        results = strategy.backtest(subset_returns)
        
        # Print key metrics
        metrics = results["performance_metrics"]
        print(f"   Total return: {metrics['total_return']:.4f} ({metrics['total_return']*100:.2f}%)")
        print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"   Max drawdown: {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
        print(f"   Win rate: {metrics['win_rate']:.4f} ({metrics['win_rate']*100:.2f}%)")
        
        # Step 5: Performance analysis
        print("\n5. Detailed performance analysis...")
        perf_analyzer = PerformanceAnalyzer(config.analysis)
        
        portfolio_returns = results["portfolio_returns"]
        performance_report = perf_analyzer.generate_performance_report(portfolio_returns)
        
        # Print detailed metrics
        return_metrics = performance_report["return_metrics"]
        risk_metrics = performance_report["risk_metrics"]
        
        print(f"   Annualized return: {return_metrics['annualized_return']:.4f} ({return_metrics['annualized_return']*100:.2f}%)")
        print(f"   Annualized volatility: {return_metrics['volatility']:.4f} ({return_metrics['volatility']*100:.2f}%)")
        print(f"   VaR (95%): {risk_metrics['var_95']:.4f} ({risk_metrics['var_95']*100:.2f}%)")
        print(f"   CVaR (95%): {risk_metrics['cvar_95']:.4f} ({risk_metrics['cvar_95']*100:.2f}%)")
        
        # Step 6: Position analysis
        print("\n6. Position analysis...")
        position_stats = strategy.get_position_statistics()
        print(f"   Average positions per day: {position_stats['average_positions_per_day']:.1f}")
        print(f"   Long position ratio: {position_stats['long_position_ratio']:.2f}")
        print(f"   Short position ratio: {position_stats['short_position_ratio']:.2f}")
        
        print("Example completed!")
        print("\nTo run the full analysis with all features:")
        print("python main.py")
        print("\nTo run with custom settings:")
        print("python main.py --env production")
        
    except Exception as e:
        print(f"\nError running example: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
